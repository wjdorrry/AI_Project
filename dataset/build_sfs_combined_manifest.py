from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BASE = ROOT / "dataset" / "sfs_semantic_filtered" / "manifest_with_manual_minimalist.csv"
WEAK = ROOT / "dataset" / "sfs_weak_filtered" / "manifest.csv"
MANUAL2_DIR = ROOT / "dataset" / "minimalist_review_2"

OUT_DIR = ROOT / "dataset" / "sfs_combined_real"
OUT_MANIFEST = OUT_DIR / "manifest.csv"

STYLE_LABELS = [
    "Elegant", "Casual", "Romantic", "Minimalist", "Sporty",
    "Streetwear", "Boho", "Office", "Evening",
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def read_csv(path: Path):
    if not path.exists():
        raise SystemExit(f"Не найден: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def main():
    rows = []
    seen = set()

    # 1) Old semantic-filtered SFS + 12 manually reviewed Minimalist.
    for r in read_csv(BASE):
        key = r["file"]
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "file": r["file"],
            "style": r["style"],
            "source_style": r.get("source_style", ""),
            "pic_name": r.get("pic_name", Path(r["file"]).name),
            "source": r.get("source", "SFS"),
            "confidence": r.get("confidence", ""),
            "margin": r.get("margin", ""),
        })

    # 2) New weak-class FashionCLIP-filtered examples.
    for r in read_csv(WEAK):
        key = r["file"]
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "file": r["file"],
            "style": r["style"],
            "source_style": r.get("source_style", ""),
            "pic_name": r.get("pic_name", Path(r["file"]).name),
            "source": r.get("source", "SFS-weak+FashionCLIP"),
            "confidence": r.get("confidence", ""),
            "margin": r.get("margin", ""),
        })

    # 3) Second batch of manually reviewed Minimalist.
    manual2 = sorted(
        p for p in MANUAL2_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ) if MANUAL2_DIR.exists() else []

    manual2_added = 0
    for p in manual2:
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        if rel in seen:
            continue
        seen.add(rel)
        rows.append({
            "file": rel,
            "style": "Minimalist",
            "source_style": "Basic",
            "pic_name": p.name,
            "source": "SFS+manual-review-2",
            "confidence": "",
            "margin": "",
        })
        manual2_added += 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUT_MANIFEST.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file", "style", "source_style", "pic_name",
                "source", "confidence", "margin",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(r["style"] for r in rows)

    print("=== SFS COMBINED REAL DATASET ===")
    print("Всего:", len(rows))
    print("Новых ручных Minimalist (batch 2):", manual2_added)
    print("По стилям:")
    for style in STYLE_LABELS:
        print(f"  {style:11s}: {counts[style]}")
    print("Manifest:", OUT_MANIFEST)

if __name__ == "__main__":
    main()
