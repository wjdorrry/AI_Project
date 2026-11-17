from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "dataset" / "week7_style_exact4"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "manifest.csv"

DATASET_ID = "lihicarmeli/fashion-stylist-multimodal-v2"
TARGETS = {
    "elegant": "Elegant",
    "romantic": "Romantic",
    "minimalist": "Minimalist",
    "boho": "Boho",
}
PER_STYLE_LIMIT = 300


def split_for(source_id: str) -> str:
    h = int(hashlib.sha1(source_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if h < 70:
        return "train"
    if h < 85:
        return "validation"
    return "test"


def main() -> None:
    counts = Counter()
    existing_ids = set()
    rows = []

    if OUT_CSV.exists():
        with OUT_CSV.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
                existing_ids.add(row["source_id"])
                counts[row["style"]] += 1

    print("Источник:", DATASET_ID)
    print("Целевые стили:", ", ".join(TARGETS.values()))
    print("Уже сохранено:", dict(counts))

    ds = load_dataset(DATASET_ID, split="train", streaming=True)

    seen = 0
    for row in ds:
        seen += 1
        raw_style = str(row.get("style_preference", "")).strip().lower()
        style = TARGETS.get(raw_style)
        if not style:
            continue

        if counts[style] >= PER_STYLE_LIMIT:
            if all(counts[s] >= PER_STYLE_LIMIT for s in TARGETS.values()):
                break
            continue

        quality = str(row.get("image_quality", "good")).strip().lower()
        if quality not in ("", "good"):
            continue

        image = row.get("image_improved")
        if image is None:
            image = row.get("image")
        if image is None:
            continue

        rid = str(row.get("id", f"row_{seen}"))
        source_id = f"{style}:{rid}"
        if source_id in existing_ids:
            continue

        style_dir = OUT_DIR / style
        style_dir.mkdir(parents=True, exist_ok=True)

        idx = counts[style] + 1
        filename = f"{style.lower()}_{idx:04d}.jpg"
        image.convert("RGB").save(style_dir / filename, "JPEG", quality=94)

        split = split_for(source_id)
        rows.append({
            "file": f"{style}/{filename}",
            "style": style,
            "split": split,
            "source": DATASET_ID,
            "source_id": source_id,
        })
        existing_ids.add(source_id)
        counts[style] += 1

        if counts[style] % 25 == 0:
            print(f"{style}: {counts[style]}")

        if all(counts[s] >= PER_STYLE_LIMIT for s in TARGETS.values()):
            break

    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "style", "split", "source", "source_id"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nDONE")
    print("Manifest:", OUT_CSV)
    for style in TARGETS.values():
        by_split = Counter(r["split"] for r in rows if r["style"] == style)
        print(
            f"{style}: {counts[style]} | "
            f"train={by_split['train']} "
            f"val={by_split['validation']} "
            f"test={by_split['test']}"
        )

    missing = {
        s: PER_STYLE_LIMIT - counts[s]
        for s in TARGETS.values()
        if counts[s] < PER_STYLE_LIMIT
    }
    if missing:
        print("\nОригинальных изображений меньше 300 для:", missing)
        print("Это нормально. Позже дополним только TRAIN аугментациями.")
    else:
        print("\nВсе 4 класса собраны по 300 оригиналов.")


if __name__ == "__main__":
    main()
