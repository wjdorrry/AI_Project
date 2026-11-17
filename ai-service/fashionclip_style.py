from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel


MODEL_ID = "patrickjohncyh/fashion-clip"
DEFAULT_TEMPERATURE = 0.05

EXPECTED_STYLES = [
    "Elegant",
    "Casual",
    "Romantic",
    "Minimalist",
    "Sporty",
    "Streetwear",
    "Boho",
    "Office",
    "Evening",
]


class FashionClipStyleClassifier:
    """Финальный классификатор стиля Week 7.

    Схема полностью совпадает с финальной оценкой:
    FashionCLIP image embedding + 75% text prototypes + 25% real-image prototypes.
    Конкретный alpha_text читается из final_prototypes.npz.
    """

    def __init__(
        self,
        prototypes_path: Path,
        *,
        device: str | torch.device | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self.prototypes_path = Path(prototypes_path)
        self.temperature = float(temperature)

        if not self.prototypes_path.exists():
            raise FileNotFoundError(f"Не найден файл прототипов: {self.prototypes_path}")

        data = np.load(self.prototypes_path, allow_pickle=True)

        self.text_prototypes = self._normalize(
            data["text_prototypes"].astype(np.float32)
        )
        self.image_prototypes = self._normalize(
            data["image_prototypes"].astype(np.float32)
        )

        raw_labels = data["style_labels"].tolist()
        self.style_labels = [str(x) for x in raw_labels]
        if self.style_labels != EXPECTED_STYLES:
            raise RuntimeError(
                "Порядок style_labels в final_prototypes.npz не совпадает "
                f"с ожидаемым: {self.style_labels}"
            )

        self.alpha_text = float(data["alpha_text"][0])
        self.alpha_image = 1.0 - self.alpha_text

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = CLIPModel.from_pretrained(MODEL_ID).to(self.device)
        self.model.eval()

        self._text_proto_tensor = torch.from_numpy(
            self.text_prototypes
        ).to(self.device)
        self._image_proto_tensor = torch.from_numpy(
            self.image_prototypes
        ).to(self.device)

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        denom = np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)
        return x / denom

    def predict(self, image: Image.Image) -> dict[str, Any]:
        enc = self.processor(images=image.convert("RGB"), return_tensors="pt")
        pixels = enc["pixel_values"].to(self.device)

        with torch.inference_mode():
            vision_out = self.model.vision_model(pixel_values=pixels)
            pooled = vision_out.pooler_output
            feat = self.model.visual_projection(pooled)
            feat = feat / feat.norm(dim=-1, keepdim=True)

            text_scores = feat @ self._text_proto_tensor.T
            image_scores = feat @ self._image_proto_tensor.T

            scores = (
                self.alpha_text * text_scores
                + self.alpha_image * image_scores
            )[0]

            probabilities = torch.softmax(
                scores / self.temperature,
                dim=0,
            )

        return {
            "labels": list(self.style_labels),
            "probabilities": probabilities.float().cpu().tolist(),
            "raw_scores": scores.float().cpu().tolist(),
            "alpha_text": self.alpha_text,
            "alpha_image": self.alpha_image,
            "device": str(self.device),
            "model_id": MODEL_ID,
        }
