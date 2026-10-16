from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


COLOR_LABELS = ["Black", "White", "Grey", "Beige", "Blue", "Brown", "Green", "Red", "Pink"]
STYLE_LABELS = ["Sportswear", "Evening", "Streetwear"]


@dataclass
class LightweightPrediction:
    color_labels: list[str]
    color_probabilities: np.ndarray
    style_labels: list[str]
    style_probabilities: np.ndarray


class LightweightImageClassifier:
    """
    Резервная обучаемая модель для локального запуска без большого model.pt.

    Она обучается на тех же dataset/images + labels_clean.csv, что и ViT.
    Это не замена основной ViT-модели, а рабочий fallback, чтобы API не
    возвращал пустой результат, когда большой checkpoint отсутствует в Git.
    """

    def __init__(self, root: Path):
        self.root = root
        self.dataset_dir = root / "dataset"
        self.images_dir = self.dataset_dir / "images"
        self.labels_csv = self.dataset_dir / "labels_clean.csv"

        self.color_model: Any | None = None
        self.style_model: Any | None = None
        self.sample_count = 0
        self.validation: dict[str, float] = {}

    @staticmethod
    def _features(img: Image.Image) -> np.ndarray:
        img = img.convert("RGB")

        # Общий вид: маленькая копия кадра.
        small = img.resize((24, 24), Image.Resampling.BILINEAR)
        arr = np.asarray(small, dtype=np.float32) / 255.0
        flat = arr.reshape(-1)

        # Общие цветовые статистики.
        mean = arr.mean(axis=(0, 1))
        std = arr.std(axis=(0, 1))

        # Центр кадра сильнее связан с одеждой, чем края/фон.
        h, w = arr.shape[:2]
        center = arr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        center_mean = center.mean(axis=(0, 1))
        center_std = center.std(axis=(0, 1))

        # Грубая информация о структуре/контрасте изображения.
        gray = arr.mean(axis=2)
        dx = np.abs(np.diff(gray, axis=1))
        dy = np.abs(np.diff(gray, axis=0))
        texture = np.array(
            [
                dx.mean(),
                dx.std(),
                dy.mean(),
                dy.std(),
            ],
            dtype=np.float32,
        )

        # Гистограммы по RGB.
        hist_parts: list[np.ndarray] = []
        for channel in range(3):
            hist, _ = np.histogram(arr[:, :, channel], bins=8, range=(0.0, 1.0), density=True)
            hist_parts.append(hist.astype(np.float32))
        hist = np.concatenate(hist_parts)

        return np.concatenate([flat, mean, std, center_mean, center_std, texture, hist]).astype(np.float32)

    def _load_dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.labels_csv.exists():
            raise RuntimeError(f"Не найден файл разметки: {self.labels_csv}")

        xs: list[np.ndarray] = []
        colors: list[str] = []
        styles: list[str] = []

        with self.labels_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_name = str(row.get("file", "")).strip()
                color = str(row.get("color", "")).strip()
                style = str(row.get("style", "")).strip()
                path = self.images_dir / file_name

                if not file_name or not path.exists():
                    continue
                if color not in COLOR_LABELS or style not in STYLE_LABELS:
                    continue

                try:
                    with Image.open(path) as img:
                        xs.append(self._features(img))
                    colors.append(color)
                    styles.append(style)
                except OSError:
                    continue

        if len(xs) < 20:
            raise RuntimeError(
                "Недостаточно изображений для резервной модели. "
                "Проверьте dataset/images и dataset/labels_clean.csv."
            )

        return np.stack(xs), np.asarray(colors), np.asarray(styles)

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

    def fit(self) -> None:
        x, y_color, y_style = self._load_dataset()
        self.sample_count = int(len(x))

        indices = np.arange(len(x))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=0.2,
            random_state=42,
            stratify=y_style,
        )

        color_eval = self._new_model()
        style_eval = self._new_model()
        color_eval.fit(x[train_idx], y_color[train_idx])
        style_eval.fit(x[train_idx], y_style[train_idx])

        color_pred = color_eval.predict(x[val_idx])
        style_pred = style_eval.predict(x[val_idx])

        self.validation = {
            "color_accuracy": float(accuracy_score(y_color[val_idx], color_pred)),
            "color_f1_macro": float(f1_score(y_color[val_idx], color_pred, average="macro", zero_division=0)),
            "style_accuracy": float(accuracy_score(y_style[val_idx], style_pred)),
            "style_f1_macro": float(f1_score(y_style[val_idx], style_pred, average="macro", zero_division=0)),
        }

        # Для рабочего ответа переобучаем на всём локальном датасете.
        self.color_model = self._new_model()
        self.style_model = self._new_model()
        self.color_model.fit(x, y_color)
        self.style_model.fit(x, y_style)

    def predict(self, img: Image.Image) -> LightweightPrediction:
        if self.color_model is None or self.style_model is None:
            self.fit()

        feat = self._features(img).reshape(1, -1)
        color_prob = self.color_model.predict_proba(feat)[0]
        style_prob = self.style_model.predict_proba(feat)[0]

        color_labels = [str(x) for x in self.color_model.classes_]
        style_labels = [str(x) for x in self.style_model.classes_]

        return LightweightPrediction(
            color_labels=color_labels,
            color_probabilities=np.asarray(color_prob, dtype=np.float32),
            style_labels=style_labels,
            style_probabilities=np.asarray(style_prob, dtype=np.float32),
        )
