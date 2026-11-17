from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_MANIFEST = ROOT / "dataset" / "sfs_semantic_filtered" / "manifest.csv"
REVIEW_DIR = ROOT / "dataset" / "minimalist_review"
OUT_MANIFEST = ROOT / "dataset" / "sfs_semantic_filtered" / "manifest_with_manual_minimalist.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def main():
    if not BASE_MANIFEST.exists():
        raise SystemExit(f"Не найден: {BASE_MANIFEST}")
    if not REVIEW_DIR.exists():
        raise SystemExit(f"Не найдена папка: {REVIEW_DIR}")

    with BASE_MANIFEST.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    existing_files = {r["file"] for r in rows}

    manual_files = sorted(
        p for p in REVIEW_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    added = 0
    for p in manual_files:
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        if rel in existing_files:
            continue
        rows.append({
            "file": rel,
            "style": "Minimalist",
            "source_style": "Basic",
            "pic_name": p.name,
            "source": "SFS+manual-review",
            "confidence": "",
            "margin": "",
        })
        existing_files.add(rel)
        added += 1

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

    print("=== MANUAL MINIMALIST ADDED ===")
    print("Ручных Minimalist:", len(manual_files))
    print("Добавлено в manifest:", added)
    print("Итого записей:", len(rows))
    print("Manifest:", OUT_MANIFEST)

if __name__ == "__main__":
    main()
