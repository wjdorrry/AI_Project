from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "sillage_style.db"
SECRET_PATH = BASE_DIR / ".auth_secret"

TOKEN_TTL_SECONDS = 60 * 60 * 24 * 7
JWT_ALGORITHM = "HS256"
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

router = APIRouter(prefix="/auth", tags=["auth"])
bearer = HTTPBearer(auto_error=False)


class RegisterRequest(BaseModel):
    display_name: str = Field(min_length=2, max_length=60)
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=1, max_length=128)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _normalize_email(value: str) -> str:
    email = value.strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Введите корректный email.",
        )
    return email


def _normalize_name(value: str) -> str:
    name = " ".join(value.strip().split())
    if len(name) < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Имя должно содержать минимум 2 символа.",
        )
    return name


def _validate_password(value: str) -> None:
    if len(value) < 8:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Пароль должен содержать минимум 8 символов.",
        )
    if not any(ch.isalpha() for ch in value) or not any(ch.isdigit() for ch in value):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Добавьте в пароль хотя бы одну букву и одну цифру.",
        )


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    n, r, p = 2**14, 8, 1
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        dklen=32,
    )
    return "$".join(
        [
            "scrypt",
            str(n),
            str(r),
            str(p),
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(digest).decode("ascii"),
        ]
    )


def _verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, n, r, p, salt_b64, digest_b64 = encoded.split("$", 5)
        if scheme != "scrypt":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
        actual = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=int(n),
            r=int(r),
            p=int(p),
            dklen=len(expected),
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _jwt_secret() -> bytes:
    env_secret = os.getenv("SILLAGE_JWT_SECRET", "").strip()
    if env_secret:
        return env_secret.encode("utf-8")

    if SECRET_PATH.exists():
        secret = SECRET_PATH.read_text(encoding="utf-8").strip()
        if secret:
            return secret.encode("utf-8")

    secret = secrets.token_urlsafe(48)
    SECRET_PATH.write_text(secret, encoding="utf-8")
    return secret.encode("utf-8")


def _create_token(user_id: int, email: str) -> str:
    now = int(time.time())
    header = {"alg": JWT_ALGORITHM, "typ": "JWT"}
    payload = {
        "sub": str(user_id),
        "email": email,
        "iat": now,
        "exp": now + TOKEN_TTL_SECONDS,
    }
    header_part = _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_part = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_part}.{payload_part}".encode("ascii")
    signature = hmac.new(_jwt_secret(), signing_input, hashlib.sha256).digest()
    return f"{header_part}.{payload_part}.{_b64url(signature)}"


def _decode_token(token: str) -> dict[str, Any]:
    try:
        header_part, payload_part, signature_part = token.split(".", 2)
        signing_input = f"{header_part}.{payload_part}".encode("ascii")
        supplied_signature = _b64url_decode(signature_part)
        expected_signature = hmac.new(_jwt_secret(), signing_input, hashlib.sha256).digest()

        if not hmac.compare_digest(supplied_signature, expected_signature):
            raise ValueError("bad signature")

        header = json.loads(_b64url_decode(header_part))
        payload = json.loads(_b64url_decode(payload_part))
        if header.get("alg") != JWT_ALGORITHM:
            raise ValueError("bad algorithm")
        if int(payload.get("exp", 0)) <= int(time.time()):
            raise ValueError("expired")
        if not payload.get("sub"):
            raise ValueError("missing subject")
        return payload
    except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Сессия недействительна или истекла. Войдите снова.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _public_user(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "display_name": row["display_name"],
        "email": row["email"],
        "created_at": row["created_at"],
    }


def _find_user_by_id(user_id: int) -> sqlite3.Row | None:
    with _connect() as conn:
        return conn.execute(
            "SELECT id, email, display_name, password_hash, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
) -> dict[str, Any]:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Сначала войдите в аккаунт.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = _decode_token(credentials.credentials)
    try:
        user_id = int(payload["sub"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Некорректная сессия.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    row = _find_user_by_id(user_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Пользователь больше не существует.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _public_user(row)


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest) -> dict[str, Any]:
    email = _normalize_email(body.email)
    display_name = _normalize_name(body.display_name)
    _validate_password(body.password)

    created_at = datetime.now(timezone.utc).isoformat()
    password_hash = _hash_password(body.password)

    try:
        with _connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (email, display_name, password_hash, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (email, display_name, password_hash, created_at),
            )
            user_id = int(cursor.lastrowid)
            conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Аккаунт с таким email уже существует.",
        )

    row = _find_user_by_id(user_id)
    assert row is not None
    user = _public_user(row)
    return {
        "access_token": _create_token(user_id, email),
        "token_type": "bearer",
        "user": user,
    }


@router.post("/login")
def login(body: LoginRequest) -> dict[str, Any]:
    email = _normalize_email(body.email)

    with _connect() as conn:
        row = conn.execute(
            "SELECT id, email, display_name, password_hash, created_at FROM users WHERE email = ? COLLATE NOCASE",
            (email,),
        ).fetchone()

    if row is None or not _verify_password(body.password, row["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный email или пароль.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = _public_user(row)
    return {
        "access_token": _create_token(int(row["id"]), row["email"]),
        "token_type": "bearer",
        "user": user,
    }


@router.get("/me")
def me(user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    return {"user": user}
