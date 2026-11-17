from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from transformers import AutoProcessor, CLIPModel

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_DIR = ROOT / "dataset" / "week7_real_holdout"
PROTO_PATH = ROOT / "ml" / "runs" / "week7_fashionclip_fewshot" / "final_prototypes.npz"
RUN_DIR = ROOT / "ml" / "runs" / "week7_fashionclip_holdout"
MODEL_ID = "patrickjohncyh/fashion-clip"

STYLE_LABELS = [
    "Elegant", "Casual", "Romantic", "Minimalist", "Sporty",
    "Streetwear", "Boho", "Office", "Evening",
]
STYLE2ID = {s: i for i, s in enumerate(STYLE_LABELS)}
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def collect_rows():
    rows = []
    for style in STYLE_LABELS:
        folder = HOLDOUT_DIR / style
        if not folder.exists():
            continue
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in EXTS:
                rows.append({
                    "path": p,
                    "file": str(p.relative_to(ROOT)).replace("\\", "/"),
                    "style": style,
                    "label": STYLE2ID[style],
                })
    return rows


def normalize(x):
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def image_embedding(model, processor, image, device):
    enc = processor(images=image, return_tensors="pt")
    pixels = enc["pixel_values"].to(device)
    with torch.no_grad():
        out = model.vision_model(pixel_values=pixels)
        pooled = out.pooler_output
        feat = model.visual_projection(pooled)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0].float().cpu().numpy()


def main():
    if not PROTO_PATH.exists():
        raise SystemExit(f"Не найдены prototypes: {PROTO_PATH}")

    rows = collect_rows()
    if not rows:
        raise SystemExit("Нет фото в dataset/week7_real_holdout/<STYLE>/")

    counts = pd.Series([r["style"] for r in rows]).value_counts()
    print("Holdout фото:", len(rows))
    print("По стилям:", counts.to_dict())

    missing = [s for s in STYLE_LABELS if counts.get(s, 0) == 0]
    if missing:
        raise SystemExit("Нет holdout фото для: " + ", ".join(missing))

    if any(counts.get(s, 0) < 5 for s in STYLE_LABELS):
        print("ВНИМАНИЕ: желательно минимум по 5 новых фото каждого стиля.")

    proto = np.load(PROTO_PATH, allow_pickle=True)
    image_proto = normalize(proto["image_prototypes"].astype(np.float32))
    text_proto = normalize(proto["text_prototypes"].astype(np.float32))
    alpha_text = float(proto["alpha_text"][0])
    alpha_image = 1.0 - alpha_text

    print(f"Смесь: text={alpha_text:.2f} image={alpha_image:.2f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Устройство:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    model.eval()

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    y_true, y_pred = [], []
    results = []

    for i, row in enumerate(rows, start=1):
        image = Image.open(row["path"]).convert("RGB")
        x = image_embedding(model, processor, image, device)
        x = normalize(x.reshape(1, -1))[0]

        text_scores = x @ text_proto.T
        image_scores = x @ image_proto.T
        scores = alpha_text * text_scores + alpha_image * image_scores

        order = np.argsort(-scores)
        pred = int(order[0])
        second = int(order[1])

        shifted = scores - scores.max()
        probs = np.exp(shifted / 0.05)
        probs = probs / probs.sum()

        conf = float(probs[pred])
        second_conf = float(probs[second])
        margin = conf - second_conf

        y_true.append(row["label"])
        y_pred.append(pred)

        results.append({
            "file": row["file"],
            "true_style": row["style"],
            "pred_style": STYLE_LABELS[pred],
            "correct": bool(pred == row["label"]),
            "confidence": conf,
            "second_style": STYLE_LABELS[second],
            "second_confidence": second_conf,
            "margin": margin,
        })

        print(
            f"[{i:02d}/{len(rows)}] {row['style']:11s} -> "
            f"{STYLE_LABELS[pred]:11s} conf={conf:.3f} margin={margin:.3f}"
        )

    pred_df = pd.DataFrame(results)
    pred_df.to_csv(RUN_DIR / "predictions.csv", index=False)

    acc = float(accuracy_score(y_true, y_pred))
    f1m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
        labels=list(range(len(STYLE_LABELS))),
        zero_division=0,
    )
    pd.DataFrame({
        "style": STYLE_LABELS,
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": support,
    }).to_csv(RUN_DIR / "per_class.csv", index=False)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(STYLE_LABELS))))
    pd.DataFrame(cm, index=STYLE_LABELS, columns=STYLE_LABELS).to_csv(
        RUN_DIR / "confusion_matrix.csv"
    )

    sid = STYLE2ID["Streetwear"]
    streetwear_fp = int(
        sum(1 for yt, yp in zip(y_true, y_pred) if yp == sid and yt != sid)
    )

    summary = {
        "model": MODEL_ID,
        "method": "FashionCLIP text + real-image prototypes",
        "samples": len(rows),
        "alpha_text": alpha_text,
        "alpha_image": alpha_image,
        "accuracy": acc,
        "f1_macro": f1m,
        "streetwear_false_positives": streetwear_fp,
        "mean_confidence": float(pred_df["confidence"].mean()),
        "mean_margin": float(pred_df["margin"].mean()),
        "counts": {s: int(counts.get(s, 0)) for s in STYLE_LABELS},
    }

    (RUN_DIR / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== FINAL REAL HOLDOUT ===")
    print(f"Samples={len(rows)}")
    print(f"Accuracy={acc:.3f}")
    print(f"Macro-F1={f1m:.3f}")
    print(f"Streetwear_FP={streetwear_fp}")
    print(f"Mean confidence={summary['mean_confidence']:.3f}")
    print(f"Mean margin={summary['mean_margin']:.3f}")
    print("Результаты:", RUN_DIR)


if __name__ == "__main__":
    main()
