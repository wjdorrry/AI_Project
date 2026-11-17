from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel

ROOT = Path(__file__).resolve().parent.parent
INPUT_MANIFEST = ROOT / "dataset" / "sfs_weak_candidates" / "manifest.csv"
PROTO_PATH = ROOT / "ml" / "runs" / "week7_fashionclip_dev72" / "final_prototypes.npz"

OUT_DIR = ROOT / "dataset" / "sfs_weak_filtered"
OUT_MANIFEST = OUT_DIR / "manifest.csv"
MINIMALIST_REVIEW = OUT_DIR / "minimalist_manual_review.csv"
ALL_PREDICTIONS = OUT_DIR / "all_predictions.csv"

MODEL_ID = "patrickjohncyh/fashion-clip"
BATCH_SIZE = 8
TEMPERATURE = 0.05

STYLE_LABELS = [
    "Elegant", "Casual", "Romantic", "Minimalist", "Sporty",
    "Streetwear", "Boho", "Office", "Evening",
]

# We only trust these semantic mappings automatically.
DIRECT_MAP = {
    "Elegant": "Elegant",
    "Urban": "Streetwear",
    "Business": "Office",
    "Business Casual": "Office",
    "Sports": "Sporty",
    "Sporty": "Sporty",
}

DEFAULT_CONF = 0.20
DEFAULT_MARGIN = 0.03

STRICT = {
    "Elegant": (0.22, 0.04),
    "Urban": (0.22, 0.04),
    "Business Casual": (0.22, 0.04),
    "Sports": (0.22, 0.04),
}


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    if not INPUT_MANIFEST.exists():
        raise SystemExit(f"Не найден: {INPUT_MANIFEST}")
    if not PROTO_PATH.exists():
        raise SystemExit(f"Не найдены prototypes: {PROTO_PATH}")

    rows = read_csv(INPUT_MANIFEST)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    proto = np.load(PROTO_PATH, allow_pickle=True)
    image_proto = normalize(proto["image_prototypes"].astype(np.float32))
    text_proto = normalize(proto["text_prototypes"].astype(np.float32))
    alpha_text = float(proto["alpha_text"][0])
    alpha_image = 1.0 - alpha_text

    print("Новых кандидатов:", len(rows))
    print(f"Смесь: text={alpha_text:.2f} image={alpha_image:.2f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Устройство:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    model.eval()

    accepted = []
    minimalist_review = []
    all_predictions = []

    accepted_counts = Counter()
    source_counts = Counter()
    disagree_counts = Counter()
    threshold_counts = Counter()

    for start in range(0, len(rows), BATCH_SIZE):
        chunk = rows[start:start + BATCH_SIZE]
        images = []
        valid_rows = []

        for row in chunk:
            path = ROOT / row["file"]
            try:
                images.append(Image.open(path).convert("RGB"))
                valid_rows.append(row)
            except Exception:
                continue

        if not valid_rows:
            continue

        enc = processor(images=images, return_tensors="pt")
        pixels = enc["pixel_values"].to(device)

        with torch.no_grad():
            out = model.vision_model(pixel_values=pixels)
            pooled = out.pooler_output
            feats = model.visual_projection(pooled)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        x = normalize(feats.float().cpu().numpy())

        for row, feat in zip(valid_rows, x):
            source_style = row["source_style"]
            source_counts[source_style] += 1

            scores = alpha_text * (feat @ text_proto.T) + alpha_image * (feat @ image_proto.T)
            order = np.argsort(-scores)
            pred_id = int(order[0])
            second_id = int(order[1])
            pred_style = STYLE_LABELS[pred_id]
            second_style = STYLE_LABELS[second_id]

            shifted = scores - scores.max()
            probs = np.exp(shifted / TEMPERATURE)
            probs = probs / probs.sum()
            confidence = float(probs[pred_id])
            second_conf = float(probs[second_id])
            margin = confidence - second_conf

            all_predictions.append({
                "file": row["file"],
                "candidate_style": row["style"],
                "source_style": source_style,
                "pred_style": pred_style,
                "second_style": second_style,
                "confidence": round(confidence, 6),
                "margin": round(margin, 6),
            })

            # Basic is never accepted automatically as Minimalist.
            if source_style == "Basic":
                if pred_style == "Minimalist":
                    minimalist_review.append({
                        "file": row["file"],
                        "source_style": source_style,
                        "pred_style": pred_style,
                        "confidence": round(confidence, 6),
                        "margin": round(margin, 6),
                        "pic_name": row["pic_name"],
                    })
                continue

            target = DIRECT_MAP.get(source_style)
            if target is None:
                continue

            if pred_style != target:
                disagree_counts[target] += 1
                continue

            min_conf, min_margin = STRICT.get(
                source_style, (DEFAULT_CONF, DEFAULT_MARGIN)
            )
            if confidence < min_conf or margin < min_margin:
                threshold_counts[target] += 1
                continue

            accepted.append({
                "file": row["file"],
                "style": target,
                "source_style": source_style,
                "pic_name": row["pic_name"],
                "source": "SFS-weak+FashionCLIP",
                "confidence": round(confidence, 6),
                "margin": round(margin, 6),
                "second_style": second_style,
            })
            accepted_counts[target] += 1

        done = min(start + BATCH_SIZE, len(rows))
        if done % 200 < BATCH_SIZE or done == len(rows):
            print(
                f"[{done}/{len(rows)}] accepted={sum(accepted_counts.values())} | "
                f"Elegant={accepted_counts['Elegant']} "
                f"Sporty={accepted_counts['Sporty']} "
                f"Streetwear={accepted_counts['Streetwear']} "
                f"Office={accepted_counts['Office']} "
                f"Minimalist-review={len(minimalist_review)}"
            )

    accepted.sort(key=lambda r: (r["style"], -float(r["confidence"])))
    minimalist_review.sort(
        key=lambda r: (-float(r["confidence"]), -float(r["margin"]))
    )

    with OUT_MANIFEST.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file", "style", "source_style", "pic_name",
                "source", "confidence", "margin", "second_style",
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

    with ALL_PREDICTIONS.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file", "candidate_style", "source_style", "pred_style",
                "second_style", "confidence", "margin",
            ],
        )
        writer.writeheader()
        writer.writerows(all_predictions)

    summary = {
        "candidates": len(rows),
        "accepted_total": len(accepted),
        "accepted_by_style": {
            s: int(accepted_counts[s])
            for s in ["Elegant", "Sporty", "Streetwear", "Office"]
        },
        "minimalist_manual_review_count": len(minimalist_review),
        "disagree": dict(disagree_counts),
        "threshold_reject": dict(threshold_counts),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== SFS WEAK FASHIONCLIP FILTER ===")
    print("Кандидатов:", len(rows))
    print("Автоматически принято:", len(accepted))
    print("  Elegant    :", accepted_counts["Elegant"])
    print("  Sporty     :", accepted_counts["Sporty"])
    print("  Streetwear :", accepted_counts["Streetwear"])
    print("  Office     :", accepted_counts["Office"])
    print("Basic→Minimalist на ручную проверку:", len(minimalist_review))
    print("Manifest:", OUT_MANIFEST)
    print("Minimalist review:", MINIMALIST_REVIEW)


if __name__ == "__main__":
    main()
