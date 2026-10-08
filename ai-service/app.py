from __future__ import annotations

import io
import json
import colorsys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import ViTImageProcessor, ViTModel

from auth import get_current_user, init_db, router as auth_router


APP_NAME = "Sillage&Style AI"

# Путь к вашей обученной модели (сгенерировано скриптом обучения).
ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "ml" / "runs" / "latest"
MODEL_PT = RUN_DIR / "model.pt"
LABELS_JSON = RUN_DIR / "label_maps.json"

DEFAULT_VIT_ID = "google/vit-base-patch16-224"

# fallback: если модель ещё не обучена/файлов нет (на всякий случай)
FALLBACK_MODE = "fallback"
TRAINED_MODE = "trained"

ALLOWED_COLORS = ["Black", "White", "Grey", "Beige", "Blue", "Brown", "Green", "Red", "Pink"]


def _center_crop(img: Image.Image, frac: float = 0.6) -> Image.Image:
    """Central crop to reduce background influence."""
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    left = (w - cw) // 2
    top = (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def _pixel_color_label(img: Image.Image) -> tuple[str, dict[str, float]]:
    """
    Rule-based color classification from pixels.
    Returns (best_label, distribution) over 9 allowed colors.
    """
    # tighter crop: меньше фона
    crop = _center_crop(img, frac=0.48).resize((96, 96))
    px = list(crop.getdata())

    # веса: центр важнее краёв (подавляем фон по краям кропа)
    wsum = 0.0
    h_sum = 0.0
    s_sum = 0.0
    v_sum = 0.0
    dark_w = 0.0
    gray_w = 0.0
    for i, (r, g, b) in enumerate(px):
        x = i % 96
        y = i // 96
        dx = (x - 47.5) / 47.5
        dy = (y - 47.5) / 47.5
        w = 1.0 / (1.0 + (dx * dx + dy * dy) * 2.0)  # ~[0.33..1]
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        wsum += w
        h_sum += h * w
        s_sum += s * w
        v_sum += v * w
        if v < 0.22:
            dark_w += w
        if s < 0.14:
            gray_w += w

    # усреднённые HSV (взвешенно)
    h = h_sum / max(wsum, 1e-9)
    s = s_sum / max(wsum, 1e-9)
    v = v_sum / max(wsum, 1e-9)
    dark_frac = dark_w / max(wsum, 1e-9)
    gray_frac = gray_w / max(wsum, 1e-9)

    # grayscale decisions
    # если большая часть пикселей тёмные — это чёрный, даже при тёплом фоне
    if dark_frac > 0.55 or v < 0.18:
        best = "Black"
    elif gray_frac > 0.65 or s < 0.12:
        best = "White" if v > 0.85 else "Grey"
    else:
        # hue-based buckets
        # red wraps around 0
        if h < 0.04 or h > 0.96:
            best = "Red"
        # pink / magenta-ish
        elif 0.86 <= h <= 0.96 and v > 0.35:
            best = "Pink"
        # green
        elif 0.25 <= h <= 0.45:
            best = "Green"
        # blue
        elif 0.55 <= h <= 0.72:
            best = "Blue"
        # brown / beige (orange-yellow hues)
        elif 0.05 <= h <= 0.18:
            best = "Brown" if v < 0.55 else "Beige"
        else:
            # fallback: choose by nearest "prototype" in HSV-ish sense
            # (simple heuristic: use beige for light warm, brown for dark warm, otherwise grey)
            if v > 0.75 and h < 0.25:
                best = "Beige"
            elif v < 0.5 and h < 0.25:
                best = "Brown"
            else:
                best = "Grey"

    # make a simple distribution: confident best + small mass to others
    eps = 0.01
    rest = (1.0 - 0.99) / (len(ALLOWED_COLORS) - 1)
    dist = {c: (0.99 if c == best else rest) for c in ALLOWED_COLORS}
    return best, dist


class MultiHeadViT(nn.Module):
    def __init__(self, vit_id: str, num_colors: int, num_styles: int):
        super().__init__()
        self.vit = ViTModel.from_pretrained(vit_id)
        hidden = self.vit.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head_color = nn.Linear(hidden, num_colors)
        self.head_style = nn.Linear(hidden, num_styles)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.vit(pixel_values=pixel_values)
        pooled = out.pooler_output
        pooled = self.dropout(pooled)
        return {
            "logits_color": self.head_color(pooled),
            "logits_style": self.head_style(pooled),
        }


app = FastAPI(title=APP_NAME)
init_db()
app.include_router(auth_router)

app.add_middleware(
    CORSMiddleware,
    # Vite может занять 5173 и запуститься на 5174/5175 и т.д.
    # Для dev-разработки разрешаем localhost/127.0.0.1 на любом порту.
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_mode: str | None = None
_vit_id: str | None = None
_processor: ViTImageProcessor | None = None
_model: MultiHeadViT | None = None
_color_labels: list[str] | None = None
_style_labels: list[str] | None = None


def _load_trained() -> None:
    global _mode, _vit_id, _processor, _model, _color_labels, _style_labels
    meta = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    _vit_id = str(meta.get("vit_id") or DEFAULT_VIT_ID)
    _color_labels = list(meta["color_labels"])
    _style_labels = list(meta["style_labels"])

    _processor = ViTImageProcessor.from_pretrained(_vit_id)
    m = MultiHeadViT(_vit_id, num_colors=len(_color_labels), num_styles=len(_style_labels))
    state = torch.load(MODEL_PT, map_location="cpu")
    m.load_state_dict(state["model"], strict=True)
    m.eval()
    _model = m
    _mode = TRAINED_MODE


def _ensure_model() -> None:
    global _mode
    if _mode is not None:
        return
    if MODEL_PT.exists() and LABELS_JSON.exists():
        _load_trained()
    else:
        _mode = FALLBACK_MODE


@app.get("/health")
def health() -> dict[str, Any]:
    _ensure_model()
    return {
        "ok": True,
        "name": APP_NAME,
        "mode": _mode,
        "model_path": str(MODEL_PT) if MODEL_PT.exists() else None,
        "labels_path": str(LABELS_JSON) if LABELS_JSON.exists() else None,
        "vit_id": _vit_id,
    }


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    _user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    _ensure_model()
    raw = await image.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    if _mode != TRAINED_MODE:
        return {
            "tags": [],
            "top": [],
            "note": "Обученная модель не найдена. Сначала обучите модель в папке ml/ (должны появиться ml/runs/latest/model.pt и label_maps.json).",
        }

    assert _processor is not None and _model is not None
    assert _color_labels is not None and _style_labels is not None

    inputs = _processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"]  # [1,3,224,224]
    with torch.no_grad():
        out = _model(pixel_values=pixel_values)
        p_color = out["logits_color"][0].softmax(dim=0)
        p_style = out["logits_style"][0].softmax(dim=0)

    items: list[dict[str, Any]] = []
    pixel_best, _pixel_dist = _pixel_color_label(img)
    for i, lab in enumerate(_color_labels):
        items.append({"tag": lab, "score": float(p_color[i])})

    for i, lab in enumerate(_style_labels):
        items.append({"tag": lab, "score": float(p_style[i])})

    items.sort(key=lambda x: x["score"], reverse=True)
    return {
        "tags": items,
        "top": items[:6],
        "note": "Стиль и цвет — обученная модель (ViT, два выхода). Пиксельная оценка только для отладки.",
        "debug": {"pixel_color": pixel_best},
    }

