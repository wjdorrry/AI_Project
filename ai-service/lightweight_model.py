from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


COLOR_LABELS = ["Black", "White", "Grey", "Beige", "Blue", "Brown", "Green", "Red", "Pink", "Burgundy", "Purple", "Yellow", "Orange", "Navy", "Cream"]
STYLE_LABELS = [
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

STYLE_ALIASES = {"Sportswear": "Sporty"}


@dataclass
class LightweightPrediction:
    color_labels: list[str]
    color_probabilities: np.ndarray
    style_labels: list[str]
    style_probabilities: np.ndarray


class LightweightImageClassifier:
    """Небольшая резервная модель для локальной проверки AI-сервиса."""

    def __init__(self, root: Path):
        self.root = root
        self.dataset_dir = root / "dataset"

        week3_labels = self.dataset_dir / "week3_labels.csv"
        week3_images = self.dataset_dir / "week3_images"
        week4_labels = self.dataset_dir / "week4_color_labels.csv"
        week4_images = self.dataset_dir / "week4_color_images"

        if week3_labels.exists() and week3_images.exists():
            self.style_labels_csv = week3_labels
            self.style_images_dir = week3_images
        else:
            self.style_labels_csv = self.dataset_dir / "labels_clean.csv"
            self.style_images_dir = self.dataset_dir / "images"

        if week4_labels.exists() and week4_images.exists():
            self.color_labels_csv = week4_labels
            self.color_images_dir = week4_images
        else:
            self.color_labels_csv = self.dataset_dir / "labels_clean.csv"
            self.color_images_dir = self.dataset_dir / "images"

        self.color_model: Any | None = None
        self.style_model: Any | None = None
        self.sample_count = 0
        self.validation: dict[str, float] = {}

    @staticmethod
    def _features(img: Image.Image) -> np.ndarray:
        img = img.convert("RGB")
        small = img.resize((24, 24), Image.Resampling.BILINEAR)
        arr = np.asarray(small, dtype=np.float32) / 255.0
        flat = arr.reshape(-1)

        mean = arr.mean(axis=(0, 1))
        std = arr.std(axis=(0, 1))

        h, w = arr.shape[:2]
        center = arr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        center_mean = center.mean(axis=(0, 1))
        center_std = center.std(axis=(0, 1))

        gray = arr.mean(axis=2)
        dx = np.abs(np.diff(gray, axis=1))
        dy = np.abs(np.diff(gray, axis=0))
        texture = np.array(
            [dx.mean(), dx.std(), dy.mean(), dy.std()],
            dtype=np.float32,
        )

        hist_parts: list[np.ndarray] = []
        for channel in range(3):
            hist, _ = np.histogram(
                arr[:, :, channel],
                bins=8,
                range=(0.0, 1.0),
                density=True,
            )
            hist_parts.append(hist.astype(np.float32))

        return np.concatenate(
            [flat, mean, std, center_mean, center_std, texture, *hist_parts]
        ).astype(np.float32)

    def _load_dataset(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_style: list[np.ndarray] = []
        y_style: list[str] = []
        x_color: list[np.ndarray] = []
        y_color: list[str] = []

        if not self.style_labels_csv.exists():
            raise RuntimeError(f"Не найден файл разметки стиля: {self.style_labels_csv}")

        with self.style_labels_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = str(row.get("file", "")).strip()
                style = STYLE_ALIASES.get(str(row.get("style", "")).strip(), str(row.get("style", "")).strip())
                path = self.style_images_dir / file_name

                if not file_name or not path.exists() or style not in STYLE_LABELS:
                    continue

                try:
                    with Image.open(path) as img:
                        feat = self._features(img)
                except OSError:
                    continue

                x_style.append(feat)
                y_style.append(style)

        if not self.color_labels_csv.exists():
            raise RuntimeError(f"Не найден файл разметки цвета: {self.color_labels_csv}")

        with self.color_labels_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = str(row.get("file", "")).strip()
                color = str(row.get("color", "")).strip()
                path = self.color_images_dir / file_name

                if not file_name or not path.exists() or color not in COLOR_LABELS:
                    continue

                try:
                    with Image.open(path) as img:
                        feat = self._features(img)
                except OSError:
                    continue

                x_color.append(feat)
                y_color.append(color)

        if len(x_style) < 20:
            raise RuntimeError("Недостаточно изображений для распознавания стиля.")
        if len(x_color) < 20:
            raise RuntimeError("Недостаточно изображений с разметкой цвета.")

        return (
            np.stack(x_style),
            np.asarray(y_style),
            np.stack(x_color),
            np.asarray(y_color),
        )

    @staticmethod
    def _new_model() -> Any:
        return make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                max_iter=2500,
                tol=1e-4,
                class_weight="balanced",
                random_state=42,
            ),
        )

    @staticmethod
    def _probability_errors(model: Any, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        probs = model.predict_proba(x)
        classes = list(model.classes_)
        one_hot = np.zeros_like(probs)
        for i, label in enumerate(y):
            one_hot[i, classes.index(label)] = 1.0
        return (
            float(mean_absolute_error(one_hot, probs)),
            float(mean_squared_error(one_hot, probs)),
        )

    def fit(self) -> None:
        x_style, y_style, x_color, y_color = self._load_dataset()
        self.sample_count = int(len(x_style) + len(x_color))

        xs_train, xs_val, ys_train, ys_val = train_test_split(
            x_style,
            y_style,
            test_size=0.2,
            random_state=42,
            stratify=y_style,
        )
        xc_train, xc_val, yc_train, yc_val = train_test_split(
            x_color,
            y_color,
            test_size=0.2,
            random_state=42,
            stratify=y_color,
        )

        style_eval = self._new_model()
        color_eval = self._new_model()
        style_eval.fit(xs_train, ys_train)
        color_eval.fit(xc_train, yc_train)

        style_pred = style_eval.predict(xs_val)
        color_pred = color_eval.predict(xc_val)
        style_mae, style_mse = self._probability_errors(style_eval, xs_val, ys_val)
        color_mae, color_mse = self._probability_errors(color_eval, xc_val, yc_val)

        self.validation = {
            "style_accuracy": float(accuracy_score(ys_val, style_pred)),
            "style_f1_macro": float(f1_score(ys_val, style_pred, average="macro", zero_division=0)),
            "style_mae": style_mae,
            "style_mse": style_mse,
            "color_accuracy": float(accuracy_score(yc_val, color_pred)),
            "color_f1_macro": float(f1_score(yc_val, color_pred, average="macro", zero_division=0)),
            "color_mae": color_mae,
            "color_mse": color_mse,
        }

        self.style_model = self._new_model()
        self.color_model = self._new_model()
        self.style_model.fit(x_style, y_style)
        self.color_model.fit(x_color, y_color)

    def predict(self, img: Image.Image) -> LightweightPrediction:
        if self.color_model is None or self.style_model is None:
            self.fit()

        feat = self._features(img).reshape(1, -1)
        color_prob = self.color_model.predict_proba(feat)[0]
        style_prob = self.style_model.predict_proba(feat)[0]

        return LightweightPrediction(
            color_labels=[str(x) for x in self.color_model.classes_],
            color_probabilities=np.asarray(color_prob, dtype=np.float32),
            style_labels=[str(x) for x in self.style_model.classes_],
            style_probabilities=np.asarray(style_prob, dtype=np.float32),
        )
