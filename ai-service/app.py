from __future__ import annotations

import colorsys
import io
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from transformers import ViTImageProcessor, ViTModel

from auth import get_current_user, init_db, router as auth_router
from lightweight_model import LightweightImageClassifier


APP_NAME = "Sillage&Style AI"

ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = ROOT / "ml" / "runs" / "latest"
MODEL_PT = RUN_DIR / "model.pt"
LABELS_JSON = RUN_DIR / "label_maps.json"

DEFAULT_VIT_ID = "google/vit-base-patch16-224"

TRAINED_MODE = "trained_vit"
LIGHTWEIGHT_MODE = "lightweight"
UNAVAILABLE_MODE = "unavailable"

CONFIDENCE_THRESHOLD = 0.55
MARGIN_THRESHOLD = 0.10

ALLOWED_COLORS = ["Black", "White", "Grey", "Beige", "Blue", "Brown", "Green", "Red", "Pink", "Burgundy", "Purple", "Yellow", "Orange", "Navy", "Cream"]
STYLE_ALIASES = {"Sportswear": "Sporty"}


def _center_crop(img: Image.Image, frac: float = 0.6) -> Image.Image:
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    left = (w - cw) // 2
    top = (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


def _pixel_color_label(img: Image.Image) -> tuple[str, dict[str, float]]:
    """Простая пиксельная оценка цвета только для диагностики."""
    crop = _center_crop(img, frac=0.48).resize((96, 96))
    px = list(crop.getdata())

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
        weight = 1.0 / (1.0 + (dx * dx + dy * dy) * 2.0)
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

        wsum += weight
        h_sum += h * weight
        s_sum += s * weight
        v_sum += v * weight
        if v < 0.22:
            dark_w += weight
        if s < 0.14:
            gray_w += weight

    h = h_sum / max(wsum, 1e-9)
    s = s_sum / max(wsum, 1e-9)
    v = v_sum / max(wsum, 1e-9)
    dark_frac = dark_w / max(wsum, 1e-9)
    gray_frac = gray_w / max(wsum, 1e-9)

    if dark_frac > 0.55 or v < 0.18:
        best = "Black"
    elif gray_frac > 0.65 or s < 0.12:
        best = "White" if v > 0.85 else "Grey"
    elif h < 0.04 or h > 0.96:
        best = "Red"
    elif 0.86 <= h <= 0.96 and v > 0.35:
        best = "Pink"
    elif 0.25 <= h <= 0.45:
        best = "Green"
    elif 0.55 <= h <= 0.72:
        best = "Blue"
    elif 0.05 <= h <= 0.18:
        best = "Brown" if v < 0.55 else "Beige"
    elif v > 0.75 and h < 0.25:
        best = "Beige"
    elif v < 0.5 and h < 0.25:
        best = "Brown"
    else:
        best = "Grey"

    rest = 0.01 / (len(ALLOWED_COLORS) - 1)
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
        pooled = self.dropout(out.pooler_output)
        return {
            "logits_color": self.head_color(pooled),
            "logits_style": self.head_style(pooled),
        }


app = FastAPI(title=APP_NAME)
init_db()
app.include_router(auth_router)

app.add_middleware(
    CORSMiddleware,
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
_lightweight: LightweightImageClassifier | None = None
_load_error: str | None = None


def _load_trained() -> None:
    global _mode, _vit_id, _processor, _model, _color_labels, _style_labels

    meta = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    _vit_id = str(meta.get("vit_id") or DEFAULT_VIT_ID)
    _color_labels = list(meta["color_labels"])
    _style_labels = list(meta["style_labels"])

    _processor = ViTImageProcessor.from_pretrained(_vit_id)
    model = MultiHeadViT(
        _vit_id,
        num_colors=len(_color_labels),
        num_styles=len(_style_labels),
    )
    state = torch.load(MODEL_PT, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    _model = model
    _mode = TRAINED_MODE


def _ensure_mode() -> None:
    global _mode, _load_error

    if _mode is not None:
        return

    if MODEL_PT.exists() and LABELS_JSON.exists():
        try:
            _load_trained()
            return
        except Exception as exc:
            _load_error = f"{type(exc).__name__}: {exc}"

    # model.pt intentionally is not stored in Git because the ViT checkpoint is large.
    # The local dataset therefore becomes the automatic fallback.
    labels_csv = ROOT / "dataset" / "labels_clean.csv"
    images_dir = ROOT / "dataset" / "images"
    if labels_csv.exists() and images_dir.exists():
        _mode = LIGHTWEIGHT_MODE
    else:
        _mode = UNAVAILABLE_MODE


def _ensure_lightweight() -> LightweightImageClassifier:
    global _lightweight

    if _lightweight is None:
        _lightweight = LightweightImageClassifier(ROOT)
        try:
            _lightweight.fit()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Не удалось подготовить резервную модель: {exc}",
            ) from exc

    return _lightweight


def _summary(labels: Sequence[str], probabilities: Sequence[float]) -> dict[str, Any]:
    ranked = sorted(
        [(str(label), float(prob)) for label, prob in zip(labels, probabilities)],
        key=lambda x: x[1],
        reverse=True,
    )

    best_label, best_conf = ranked[0]
    second_label, second_conf = ranked[1] if len(ranked) > 1 else ("", 0.0)
    margin = best_conf - second_conf

    return {
        "label": best_label,
        "confidence": best_conf,
        "second_label": second_label,
        "second_confidence": second_conf,
        "margin": margin,
        "low_confidence": best_conf < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD,
    }


def _items(
    color_labels: Sequence[str],
    color_probabilities: Sequence[float],
    style_labels: Sequence[str],
    style_probabilities: Sequence[float],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for label, prob in zip(color_labels, color_probabilities):
        result.append({"tag": str(label), "score": float(prob), "category": "color"})

    for label, prob in zip(style_labels, style_probabilities):
        result.append({"tag": str(label), "score": float(prob), "category": "style"})

    result.sort(key=lambda x: x["score"], reverse=True)
    return result


@app.get("/health")
def health() -> dict[str, Any]:
    _ensure_mode()

    lightweight_info: dict[str, Any] | None = None
    if _lightweight is not None:
        lightweight_info = {
            "sample_count": _lightweight.sample_count,
            "validation": _lightweight.validation,
        }

    return {
        "ok": _mode != UNAVAILABLE_MODE,
        "name": APP_NAME,
        "mode": _mode,
        "trained_weights_found": MODEL_PT.exists(),
        "model_path": str(MODEL_PT) if MODEL_PT.exists() else None,
        "labels_path": str(LABELS_JSON) if LABELS_JSON.exists() else None,
        "vit_id": _vit_id,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "margin_threshold": MARGIN_THRESHOLD,
        "load_error": _load_error,
        "lightweight": lightweight_info,
    }


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    _user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    _ensure_mode()

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Файл изображения пуст.")

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Не удалось прочитать изображение.") from exc

    pixel_best, _pixel_dist = _pixel_color_label(img)

    if _mode == TRAINED_MODE:
        assert _processor is not None and _model is not None
        assert _color_labels is not None and _style_labels is not None

        inputs = _processor(images=img, return_tensors="pt")
        with torch.no_grad():
            out = _model(pixel_values=inputs["pixel_values"])
            color_prob = out["logits_color"][0].softmax(dim=0).cpu().tolist()
            style_prob = out["logits_style"][0].softmax(dim=0).cpu().tolist()

        color_labels = _color_labels
        style_labels = [STYLE_ALIASES.get(x, x) for x in _style_labels]
        note = "Результат получен обученной multi-task ViT-моделью."

    elif _mode == LIGHTWEIGHT_MODE:
        fallback = _ensure_lightweight()
        pred = fallback.predict(img)

        color_labels = pred.color_labels
        color_prob = pred.color_probabilities.tolist()
        style_labels = pred.style_labels
        style_prob = pred.style_probabilities.tolist()
        note = (
            "Большой checkpoint ViT model.pt отсутствует в локальной копии. "
            "Использована резервная модель, обученная на dataset/labels_clean.csv и dataset/images."
        )

    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "AI-модель недоступна: нет model.pt и не найден локальный датасет "
                "для резервного классификатора."
            ),
        )

    color_summary = _summary(color_labels, color_prob)
    style_summary = _summary(style_labels, style_prob)
    items = _items(color_labels, color_prob, style_labels, style_prob)

    low_confidence = bool(
        color_summary["low_confidence"] or style_summary["low_confidence"]
    )

    response: dict[str, Any] = {
        "model_mode": _mode,
        "tags": items,
        "top": items[:6],
        "prediction": {
            "color": color_summary,
            "style": style_summary,
            "low_confidence": low_confidence,
        },
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "margin_threshold": MARGIN_THRESHOLD,
        "note": note,
        "debug": {"pixel_color": pixel_best},
    }

    if _mode == LIGHTWEIGHT_MODE and _lightweight is not None:
        response["validation"] = _lightweight.validation
        response["training_samples"] = _lightweight.sample_count

    return response
