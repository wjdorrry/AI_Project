from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parent.parent
OUT_IMAGES = ROOT / "dataset" / "week4_color_images"
OUT_LABELS = ROOT / "dataset" / "week4_color_labels.csv"

HF_DATASET = "HEBA2002/fashion-product-images-small"

COLOR_MAP = {
    "Black": "Black",
    "White": "White",
    "Grey": "Grey",
    "Beige": "Beige",
    "Blue": "Blue",
    "Brown": "Brown",
    "Green": "Green",
    "Red": "Red",
    "Pink": "Pink",
    "Burgundy": "Burgundy",
    "Maroon": "Burgundy",
    "Purple": "Purple",
    "Yellow": "Yellow",
    "Orange": "Orange",
    "Navy Blue": "Navy",
    "Cream": "Cream",
}

COLOR_LABELS = list(dict.fromkeys(COLOR_MAP.values()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-color", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if OUT_IMAGES.exists():
        shutil.rmtree(OUT_IMAGES)
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)

    print(f"Загрузка внешнего датасета: {HF_DATASET}")
    print("Используются только изображения одежды (masterCategory = Apparel).")

    ds = load_dataset(HF_DATASET, split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=10000)

    selected: dict[str, list[dict]] = {label: [] for label in COLOR_LABELS}

    for row in ds:
        if str(row.get("masterCategory", "")).strip() != "Apparel":
            continue

        source_color = str(row.get("baseColour", "")).strip()
        target_color = COLOR_MAP.get(source_color)
        if target_color is None:
            continue

        if len(selected[target_color]) >= args.per_color:
            continue

        image = row.get("image")
        if image is None:
            continue

        selected[target_color].append(row)

        if all(len(items) >= args.per_color for items in selected.values()):
            break

    missing = {
        label: args.per_color - len(items)
        for label, items in selected.items()
        if len(items) < args.per_color
    }
    if missing:
        details = ", ".join(f"{k}: не хватает {v}" for k, v in missing.items())
        raise RuntimeError(
            "Не удалось собрать сбалансированный набор только из категории Apparel. "
            f"{details}"
        )

    rows: list[dict[str, str]] = []
    rng = random.Random(args.seed)

    for color in COLOR_LABELS:
        items = selected[color]
        rng.shuffle(items)

        for i, row in enumerate(items, start=1):
            image = row["image"].convert("RGB")
            dst_name = f"{color.lower().replace(' ', '_')}_{i:03d}.jpg"
            image.save(OUT_IMAGES / dst_name, format="JPEG", quality=95)

            rows.append(
                {
                    "file": dst_name,
                    "color": color,
                    "style": "",
                    "source": HF_DATASET,
                    "source_color": str(row.get("baseColour", "")).strip(),
                    "product": str(row.get("productDisplayName", "")).strip(),
                }
            )

    rng.shuffle(rows)

    with OUT_LABELS.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "color", "style", "source", "source_color", "product"],
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["color"] for row in rows)

    print("\nГотово.")
    print(f"Файл разметки: {OUT_LABELS}")
    print(f"Изображения: {OUT_IMAGES}")
    print(f"Всего изображений: {len(rows)}")
    for color in COLOR_LABELS:
        print(f"  {color}: {counts[color]}")


if __name__ == "__main__":
    main()
