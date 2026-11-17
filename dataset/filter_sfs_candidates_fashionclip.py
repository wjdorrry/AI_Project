from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, CLIPModel

ROOT = Path(__file__).resolve().parent.parent
CANDIDATE_MANIFEST = ROOT / "dataset" / "sfs_needed_candidates" / "manifest.csv"
PROTO_PATH = ROOT / "ml" / "runs" / "week7_fashionclip_dev72" / "final_prototypes.npz"
OUT_DIR = ROOT / "dataset" / "sfs_filtered_real"
OUT_MANIFEST = OUT_DIR / "manifest.csv"
OUT_ALL = OUT_DIR / "all_predictions.csv"

MODEL_ID = "patrickjohncyh/fashion-clip"
BATCH_SIZE = int(os.getenv("SFS_FILTER_BATCH_SIZE", "8"))

STYLE_LABELS = [
    "Elegant", "Casual", "Romantic", "Minimalist", "Sporty",
    "Streetwear", "Boho", "Office", "Evening",
]
STYLE2ID = {s: i for i, s in enumerate(STYLE_LABELS)}

# Normal threshold for fairly direct SFS -> our-style mappings.
MIN_CONF = 0.25
MIN_MARGIN = 0.05

# These source labels are broader than our target label, so filter them harder.
AMBIGUOUS_SOURCE_STYLES = {"Basic", "Urban", "Business Casual"}
AMBIG_MIN_CONF = 0.30
AMBIG_MIN_MARGIN = 0.08

TEMPERATURE = 0.05


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def load_rows() -> list[dict[str, str]]:
    if not CANDIDATE_MANIFEST.exists():
        raise SystemExit(f"Не найден manifest: {CANDIDATE_MANIFEST}")
    with CANDIDATE_MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    rows = load_rows()
    if not PROTO_PATH.exists():
        raise SystemExit(f"Не найдены dev72 prototypes: {PROTO_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    proto = np.load(PROTO_PATH, allow_pickle=True)
    image_proto = normalize(proto["image_prototypes"].astype(np.float32))
    text_proto = normalize(proto["text_prototypes"].astype(np.float32))
    alpha_text = float(proto["alpha_text"][0])
    alpha_image = 1.0 - alpha_text

    print("Кандидатов:", len(rows))
    print(f"Смесь FashionCLIP: text={alpha_text:.2f} image={alpha_image:.2f}")
    print(
        f"Обычный фильтр: conf>={MIN_CONF:.2f}, margin>={MIN_MARGIN:.2f} | "
        f"неоднозначные: conf>={AMBIG_MIN_CONF:.2f}, margin>={AMBIG_MIN_MARGIN:.2f}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Устройство:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    model.eval()

    accepted_rows: list[dict[str, str | float]] = []
    prediction_rows: list[dict[str, str | float | bool]] = []
    accepted_counts = Counter()
    source_counts = Counter()
    rejected_disagree = Counter()
    rejected_threshold = Counter()
    bad_files = 0

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
                bad_files += 1

        if not valid_rows:
            continue

        enc = processor(images=images, return_tensors="pt")
        pixels = enc["pixel_values"].to(device)

        with torch.no_grad():
            out = model.vision_model(pixel_values=pixels)
            pooled = out.pooler_output
            feats = model.visual_projection(pooled)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        x = feats.float().cpu().numpy()
        x = normalize(x)

        for row, feat in zip(valid_rows, x):
            intended = row["style"]
            source_style = row["source_style"]
            source_counts[intended] += 1

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

            ambiguous = source_style in AMBIGUOUS_SOURCE_STYLES
            min_conf = AMBIG_MIN_CONF if ambiguous else MIN_CONF
            min_margin = AMBIG_MIN_MARGIN if ambiguous else MIN_MARGIN

            agrees = pred_style == intended
            passes_threshold = confidence >= min_conf and margin >= min_margin
            accepted = agrees and passes_threshold

            if accepted:
                accepted_counts[intended] += 1
                accepted_rows.append({
                    "file": row["file"],
                    "style": intended,
                    "source_style": source_style,
                    "pic_name": row["pic_name"],
                    "source": "SFS+FashionCLIP",
                    "confidence": round(confidence, 6),
                    "margin": round(margin, 6),
                    "second_style": second_style,
                })
            elif not agrees:
                rejected_disagree[intended] += 1
            else:
                rejected_threshold[intended] += 1

            prediction_rows.append({
                "file": row["file"],
                "candidate_style": intended,
                "source_style": source_style,
                "pred_style": pred_style,
                "second_style": second_style,
                "confidence": round(confidence, 6),
                "margin": round(margin, 6),
                "accepted": accepted,
            })

        done = min(start + BATCH_SIZE, len(rows))
        if done % 200 < BATCH_SIZE or done == len(rows):
            print(
                f"[{done}/{len(rows)}] accepted={sum(accepted_counts.values())} | "
                + ", ".join(f"{s}={accepted_counts[s]}" for s in STYLE_LABELS)
            )

    with OUT_MANIFEST.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file", "style", "source_style", "pic_name", "source",
                "confidence", "margin", "second_style",
            ],
        )
        writer.writeheader()
        writer.writerows(accepted_rows)

    with OUT_ALL.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file", "candidate_style", "source_style", "pred_style",
                "second_style", "confidence", "margin", "accepted",
            ],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    summary = {
        "candidates": len(rows),
        "accepted": int(sum(accepted_counts.values())),
        "alpha_text": alpha_text,
        "alpha_image": alpha_image,
        "thresholds": {
            "normal_conf": MIN_CONF,
            "normal_margin": MIN_MARGIN,
            "ambiguous_conf": AMBIG_MIN_CONF,
            "ambiguous_margin": AMBIG_MIN_MARGIN,
        },
        "accepted_by_style": {s: int(accepted_counts[s]) for s in STYLE_LABELS},
        "candidate_by_style": {s: int(source_counts[s]) for s in STYLE_LABELS},
        "rejected_disagreement": {s: int(rejected_disagree[s]) for s in STYLE_LABELS},
        "rejected_threshold": {s: int(rejected_threshold[s]) for s in STYLE_LABELS},
        "bad_files": bad_files,
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== SFS FASHIONCLIP FILTER ===")
    print("Кандидатов:", len(rows))
    print("Принято:", sum(accepted_counts.values()))
    print("По стилям:")
    for style in STYLE_LABELS:
        print(
            f"  {style:11s}: {accepted_counts[style]:3d} / {source_counts[style]:3d}"
            f" | disagree={rejected_disagree[style]:3d}"
            f" threshold={rejected_threshold[style]:3d}"
        )
    print("Плохих файлов:", bad_files)
    print("Manifest:", OUT_MANIFEST)
    print("Все предсказания:", OUT_ALL)
    print(
        "\nВАЖНО: файлы не копировались — фильтр создал только manifest. "
        "Поэтому дополнительное место почти не занято."
    )


if __name__ == "__main__":
    main()
