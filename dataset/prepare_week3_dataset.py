from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parent.parent
LOCAL_IMAGES = ROOT / "dataset" / "images"
LOCAL_LABELS = ROOT / "dataset" / "labels_clean.csv"
OUT_IMAGES = ROOT / "dataset" / "week3_images"
OUT_LABELS = ROOT / "dataset" / "week3_labels.csv"

HF_DATASET = "lihicarmeli/fashion-stylist-multimodal-v2"

# Для 3-й недели используем 9 понятных визуальных классов.
# Три старых класса берём из собственного датасета, ещё шесть — из открытого
# MIT-лицензированного fashion-датасета.
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

LOCAL_STYLE_MAP = {
    "Sportswear": "Sporty",
    "Streetwear": "Streetwear",
    "Evening": "Evening",
}

EXTERNAL_STYLE_MAP = {
    "elegant": "Elegant",
    "casual": "Casual",
    "romantic": "Romantic",
    "minimalist": "Minimalist",
    "boho": "Boho",
    "office": "Office",
}


def read_local_rows() -> list[dict[str, str]]:
    with LOCAL_LABELS.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def prepare_local(per_style: int, rng: random.Random) -> list[dict[str, str]]:
    rows = read_local_rows()
    by_style: dict[str, list[dict[str, str]]] = {s: [] for s in LOCAL_STYLE_MAP}

    for row in rows:
        style = str(row.get("style", "")).strip()
        file_name = str(row.get("file", "")).strip()
        if style in by_style and (LOCAL_IMAGES / file_name).exists():
            by_style[style].append(row)

    output: list[dict[str, str]] = []
    for old_style, new_style in LOCAL_STYLE_MAP.items():
        candidates = by_style[old_style]
        rng.shuffle(candidates)
        if len(candidates) < per_style:
            raise RuntimeError(
                f"Для класса {old_style} найдено только {len(candidates)} изображений; "
                f"нужно не меньше {per_style}."
            )

        for i, row in enumerate(candidates[:per_style], start=1):
            src = LOCAL_IMAGES / row["file"]
            dst_name = f"local_{new_style.lower()}_{i:03d}{src.suffix.lower()}"
            shutil.copy2(src, OUT_IMAGES / dst_name)
            output.append(
                {
                    "file": dst_name,
                    "color": str(row.get("color", "")).strip(),
                    "style": new_style,
                    "source": "local",
                }
            )

    return output


def prepare_external(per_style: int, rng: random.Random) -> list[dict[str, str]]:
    print(f"Загрузка внешнего датасета: {HF_DATASET}")
    ds = load_dataset(HF_DATASET, split="train")

    by_style: dict[str, list[int]] = {s: [] for s in EXTERNAL_STYLE_MAP}
    for idx, value in enumerate(ds["style_preference"]):
        key = str(value).strip().lower()
        quality = str(ds["image_quality"][idx]).strip().lower()
        if key in by_style and quality == "good":
            by_style[key].append(idx)

    output: list[dict[str, str]] = []
    for old_style, new_style in EXTERNAL_STYLE_MAP.items():
        indices = by_style[old_style]
        rng.shuffle(indices)
        if len(indices) < per_style:
            raise RuntimeError(
                f"Во внешнем датасете для класса {old_style} найдено только {len(indices)} "
                f"изображений; нужно не меньше {per_style}."
            )

        for i, idx in enumerate(indices[:per_style], start=1):
            row = ds[int(idx)]
            image = row.get("image_improved") or row.get("image_original")
            if image is None:
                raise RuntimeError(f"У строки {idx} отсутствует изображение.")

            dst_name = f"hf_{new_style.lower()}_{i:03d}.jpg"
            image.convert("RGB").save(OUT_IMAGES / dst_name, format="JPEG", quality=92)

            # Для внешних изображений цвет не используем как обучающую метку:
            # их primary_color не совпадает с 9 цветовыми классами нашего проекта.
            output.append(
                {
                    "file": dst_name,
                    "color": "",
                    "style": new_style,
                    "source": HF_DATASET,
                }
            )

    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-style", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if OUT_IMAGES.exists():
        shutil.rmtree(OUT_IMAGES)
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(prepare_local(args.per_style, rng))
    rows.extend(prepare_external(args.per_style, rng))
    rng.shuffle(rows)

    with OUT_LABELS.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "color", "style", "source"])
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["style"] for row in rows)
    print("\nГотово.")
    print(f"Файл разметки: {OUT_LABELS}")
    print(f"Изображения: {OUT_IMAGES}")
    print(f"Всего изображений: {len(rows)}")
    for style in STYLE_LABELS:
        print(f"  {style}: {counts[style]}")


if __name__ == "__main__":
    main()
