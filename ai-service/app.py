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
from fashionclip_style import FashionClipStyleClassifier
from lightweight_model import LightweightImageClassifier


APP_NAME = "Sillage&Style AI"

ROOT = Path(__file__).resolve().parent.parent

# Финальная связка моделей:
#   цвет  -> Week 6 ViT на 15 цветах;
#   стиль -> Week 7 FashionCLIP + text/image prototypes.
COLOR_RUN_DIR = ROOT / "ml" / "runs" / "week6_color_7500"
COLOR_MODEL_PT = COLOR_RUN_DIR / "model.pt"

STYLE_RUN_DIR = ROOT / "ml" / "runs" / "week7_fashionclip_dev72"
STYLE_PROTOTYPES_NPZ = STYLE_RUN_DIR / "final_prototypes.npz"

# Старый путь оставлен только как fallback для старых локальных копий.
LEGACY_RUN_DIR = ROOT / "ml" / "runs" / "latest"
LEGACY_MODEL_PT = LEGACY_RUN_DIR / "model.pt"
LEGACY_LABELS_JSON = LEGACY_RUN_DIR / "label_maps.json"

DEFAULT_VIT_ID = "google/vit-base-patch16-224"

HYBRID_MODE = "hybrid_vit_fashionclip"
TRAINED_MODE = "trained_vit"
LIGHTWEIGHT_MODE = "lightweight"
UNAVAILABLE_MODE = "unavailable"

COLOR_CONFIDENCE_THRESHOLD = 0.30
COLOR_MARGIN_THRESHOLD = 0.08

STYLE_CONFIDENCE_THRESHOLD = 0.20
STYLE_MARGIN_THRESHOLD = 0.03

COLOR_LABELS = [
    "Black", "White", "Grey", "Beige", "Blue", "Brown", "Green", "Red", "Pink",
    "Burgundy", "Purple", "Yellow", "Orange", "Navy", "Cream",
]
STYLE_LABELS = [
    "Elegant", "Casual", "Romantic", "Minimalist", "Sporty",
    "Streetwear", "Boho", "Office", "Evening",
]
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

    rest = 0.01 / (len(COLOR_LABELS) - 1)
    dist = {c: (0.99 if c == best else rest) for c in COLOR_LABELS}
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
_style_classifier: FashionClipStyleClassifier | None = None
_lightweight: LightweightImageClassifier | None = None
_load_error: str | None = None


def _load_week6_color() -> None:
    global _vit_id, _processor, _model, _color_labels, _style_labels

    _vit_id = DEFAULT_VIT_ID
    _color_labels = list(COLOR_LABELS)
    _style_labels = list(STYLE_LABELS)

    _processor = ViTImageProcessor.from_pretrained(_vit_id)
    model = MultiHeadViT(
        _vit_id,
        num_colors=len(_color_labels),
        num_styles=len(_style_labels),
    )

    checkpoint = torch.load(
        COLOR_MODEL_PT,
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()

    _model = model


def _load_final_style() -> None:
    global _style_classifier

    _style_classifier = FashionClipStyleClassifier(
        STYLE_PROTOTYPES_NPZ,
        device=("cuda" if torch.cuda.is_available() else "cpu"),
    )


def _load_legacy_trained() -> None:
    global _mode, _vit_id, _processor, _model, _color_labels, _style_labels

    meta = json.loads(LEGACY_LABELS_JSON.read_text(encoding="utf-8"))
    _vit_id = str(meta.get("vit_id") or DEFAULT_VIT_ID)
    _color_labels = list(meta["color_labels"])
    _style_labels = list(meta["style_labels"])

    _processor = ViTImageProcessor.from_pretrained(_vit_id)
    model = MultiHeadViT(
        _vit_id,
        num_colors=len(_color_labels),
        num_styles=len(_style_labels),
    )
    state = torch.load(LEGACY_MODEL_PT, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    model.load_state_dict(state, strict=True)
    model.eval()

    _model = model
    _mode = TRAINED_MODE


def _ensure_mode() -> None:
    global _mode, _load_error

    if _mode is not None:
        return

    # Основной финальный режим проекта.
    if COLOR_MODEL_PT.exists() and STYLE_PROTOTYPES_NPZ.exists():
        try:
            _load_week6_color()
            _load_final_style()
            _mode = HYBRID_MODE
            return
        except Exception as exc:
            _load_error = f"Hybrid load failed: {type(exc).__name__}: {exc}"

    # Старый ViT оставляем только как fallback.
    if LEGACY_MODEL_PT.exists() and LEGACY_LABELS_JSON.exists():
        try:
            _load_legacy_trained()
            return
        except Exception as exc:
            legacy_error = f"Legacy load failed: {type(exc).__name__}: {exc}"
            _load_error = (
                f"{_load_error}; {legacy_error}"
                if _load_error
                else legacy_error
            )

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


def _summary(
    labels: Sequence[str],
    probabilities: Sequence[float],
    *,
    confidence_threshold: float,
    margin_threshold: float,
) -> dict[str, Any]:
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
        "low_confidence": best_conf < confidence_threshold or margin < margin_threshold,
        "confidence_threshold": confidence_threshold,
        "margin_threshold": margin_threshold,
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

    style_info: dict[str, Any] | None = None
    if _style_classifier is not None:
        style_info = {
            "model_id": _style_classifier.model.model_type
            if hasattr(_style_classifier.model, "model_type")
            else "patrickjohncyh/fashion-clip",
            "prototype_path": str(STYLE_PROTOTYPES_NPZ),
            "alpha_text": _style_classifier.alpha_text,
            "alpha_image": _style_classifier.alpha_image,
            "device": str(_style_classifier.device),
            "final_holdout_accuracy": 0.889,
            "final_holdout_macro_f1": 0.886,
        }

    return {
        "ok": _mode != UNAVAILABLE_MODE,
        "name": APP_NAME,
        "mode": _mode,
        "color_model_path": str(COLOR_MODEL_PT) if COLOR_MODEL_PT.exists() else None,
        "style_prototypes_path": (
            str(STYLE_PROTOTYPES_NPZ)
            if STYLE_PROTOTYPES_NPZ.exists()
            else None
        ),
        "vit_id": _vit_id,
        "thresholds": {
            "color": {
                "confidence": COLOR_CONFIDENCE_THRESHOLD,
                "margin": COLOR_MARGIN_THRESHOLD,
            },
            "style": {
                "confidence": STYLE_CONFIDENCE_THRESHOLD,
                "margin": STYLE_MARGIN_THRESHOLD,
            },
        },
        "load_error": _load_error,
        "style_model": style_info,
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

    if _mode == HYBRID_MODE:
        assert _processor is not None and _model is not None
        assert _color_labels is not None and _style_classifier is not None

        inputs = _processor(images=img, return_tensors="pt")
        with torch.inference_mode():
            out = _model(pixel_values=inputs["pixel_values"])
            color_prob = out["logits_color"][0].softmax(dim=0).cpu().tolist()

        style_pred = _style_classifier.predict(img)

        color_labels = _color_labels
        style_labels = style_pred["labels"]
        style_prob = style_pred["probabilities"]
        note = (
            "Цвет: Week 6 multi-task ViT на 15 цветах. "
            "Стиль: финальный Week 7 FashionCLIP "
            f"(text={style_pred['alpha_text']:.2f}, "
            f"real-image={style_pred['alpha_image']:.2f})."
        )

    elif _mode == TRAINED_MODE:
        assert _processor is not None and _model is not None
        assert _color_labels is not None and _style_labels is not None

        inputs = _processor(images=img, return_tensors="pt")
        with torch.inference_mode():
            out = _model(pixel_values=inputs["pixel_values"])
            color_prob = out["logits_color"][0].softmax(dim=0).cpu().tolist()
            style_prob = out["logits_style"][0].softmax(dim=0).cpu().tolist()

        color_labels = _color_labels
        style_labels = [STYLE_ALIASES.get(x, x) for x in _style_labels]
        note = "Использован старый ViT fallback."

    elif _mode == LIGHTWEIGHT_MODE:
        fallback = _ensure_lightweight()
        pred = fallback.predict(img)

        color_labels = pred.color_labels
        color_prob = pred.color_probabilities.tolist()
        style_labels = pred.style_labels
        style_prob = pred.style_probabilities.tolist()
        note = (
            "Финальные checkpoint/prototypes недоступны. "
            "Использована резервная локальная модель."
        )

    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "AI-модель недоступна: не найдены финальные локальные модели "
                "и нет датасета для резервного классификатора."
            ),
        )

    color_summary = _summary(
        color_labels,
        color_prob,
        confidence_threshold=COLOR_CONFIDENCE_THRESHOLD,
        margin_threshold=COLOR_MARGIN_THRESHOLD,
    )
    style_summary = _summary(
        style_labels,
        style_prob,
        confidence_threshold=STYLE_CONFIDENCE_THRESHOLD,
        margin_threshold=STYLE_MARGIN_THRESHOLD,
    )
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
        "thresholds": {
            "color": {
                "confidence": COLOR_CONFIDENCE_THRESHOLD,
                "margin": COLOR_MARGIN_THRESHOLD,
            },
            "style": {
                "confidence": STYLE_CONFIDENCE_THRESHOLD,
                "margin": STYLE_MARGIN_THRESHOLD,
            },
        },
        "note": note,
        "debug": {"pixel_color": pixel_best},
    }

    if _mode == HYBRID_MODE and _style_classifier is not None:
        response["style_model"] = {
            "name": "FashionCLIP",
            "alpha_text": _style_classifier.alpha_text,
            "alpha_image": _style_classifier.alpha_image,
            "final_holdout_accuracy": 0.889,
            "final_holdout_macro_f1": 0.886,
        }

    if _mode == LIGHTWEIGHT_MODE and _lightweight is not None:
        response["validation"] = _lightweight.validation
        response["training_samples"] = _lightweight.sample_count

    return response
