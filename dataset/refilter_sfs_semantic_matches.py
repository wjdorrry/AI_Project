from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS = ROOT / "dataset" / "sfs_filtered_real" / "all_predictions.csv"
CANDIDATES = ROOT / "dataset" / "sfs_needed_candidates" / "manifest.csv"
OUT_DIR = ROOT / "dataset" / "sfs_semantic_filtered"
OUT_MANIFEST = OUT_DIR / "manifest.csv"
MINIMALIST_REVIEW = OUT_DIR / "minimalist_manual_review.csv"

# Only mappings that are semantically close enough to trust automatically.
DIRECT_MAP = {
    "Casual": "Casual",
    "Romantic": "Romantic",
    "Bohemian": "Boho",
    "Urban": "Streetwear",
    "Business": "Office",
    "Business Casual": "Office",
    "Formal": "Evening",
    "Sports": "Sporty",
    "Sporty": "Sporty",
    "Elegant": "Elegant",
}

# Basic is intentionally excluded. It is not the same as Minimalist.
# We only export Basic -> Minimalist matches for manual review.

# Slightly relaxed thresholds because source label + FashionCLIP must agree.
DEFAULT_CONF = 0.20
DEFAULT_MARGIN = 0.03

# Broader source labels need a bit more separation.
STRICT = {
    "Urban": (0.22, 0.04),
    "Business Casual": (0.22, 0.04),
    "Elegant": (0.22, 0.04),
    "Sports": (0.22, 0.04),
}

def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def main():
    if not PREDICTIONS.exists():
        raise SystemExit(f"Не найден: {PREDICTIONS}")
    if not CANDIDATES.exists():
        raise SystemExit(f"Не найден: {CANDIDATES}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    preds = read_csv(PREDICTIONS)
    cand = read_csv(CANDIDATES)
    by_file = {r["file"]: r for r in cand}

    accepted = []
    rejected = Counter()
    counts = Counter()
    raw_agree = Counter()
    minimalist_review = []

    for row in preds:
        source_style = row["source_style"]
        pred_style = row["pred_style"]
        conf = float(row["confidence"])
        margin = float(row["margin"])

        # Keep potential Minimalist examples separate for human review.
        if source_style == "Basic" and pred_style == "Minimalist":
            base = by_file.get(row["file"], {})
            minimalist_review.append({
                "file": row["file"],
                "source_style": source_style,
                "pred_style": pred_style,
                "confidence": conf,
                "margin": margin,
                "pic_name": base.get("pic_name", Path(row["file"]).name),
            })
            continue

        target = DIRECT_MAP.get(source_style)
        if target is None:
            continue
        if pred_style != target:
            rejected[f"{source_style}:disagree"] += 1
            continue

        raw_agree[target] += 1
        min_conf, min_margin = STRICT.get(source_style, (DEFAULT_CONF, DEFAULT_MARGIN))
        if conf < min_conf or margin < min_margin:
            rejected[f"{source_style}:threshold"] += 1
            continue

        base = by_file.get(row["file"], {})
        accepted.append({
            "file": row["file"],
            "style": target,
            "source_style": source_style,
            "pic_name": base.get("pic_name", Path(row["file"]).name),
            "source": "SFS+FashionCLIP-semantic",
            "confidence": conf,
            "margin": margin,
        })
        counts[target] += 1

    accepted.sort(key=lambda r: (r["style"], -float(r["confidence"]), -float(r["margin"])))
    minimalist_review.sort(key=lambda r: (-float(r["confidence"]), -float(r["margin"])))

    with OUT_MANIFEST.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file", "style", "source_style", "pic_name",
                "source", "confidence", "margin",
            ],
        )
        writer.writeheader()
        writer.writerows(accepted)

    with MINIMALIST_REVIEW.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file", "source_style", "pred_style",
                "confidence", "margin", "pic_name",
            ],
        )
        writer.writeheader()
        writer.writerows(minimalist_review)

    summary = {
        "accepted_total": len(accepted),
        "accepted_by_style": {k: int(counts[k]) for k in [
            "Elegant","Casual","Romantic","Minimalist","Sporty",
            "Streetwear","Boho","Office","Evening"
        ]},
        "raw_agreement_before_threshold": dict(raw_agree),
        "minimalist_manual_review_count": len(minimalist_review),
        "thresholds": {
            "default_conf": DEFAULT_CONF,
            "default_margin": DEFAULT_MARGIN,
            "strict": STRICT,
        },
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== SFS SEMANTIC REFILTER ===")
    print("Принято:", len(accepted))
    print("По стилям:")
    for style in [
        "Elegant","Casual","Romantic","Minimalist","Sporty",
        "Streetwear","Boho","Office","Evening"
    ]:
        print(f"  {style:11s}: {counts[style]}")
    print("Basic→Minimalist на ручную проверку:", len(minimalist_review))
    print("Manifest:", OUT_MANIFEST)
    print("Minimalist review:", MINIMALIST_REVIEW)

if __name__ == "__main__":
    main()
