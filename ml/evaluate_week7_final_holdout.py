from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import AutoProcessor, CLIPModel

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_DIR = ROOT / "dataset" / "week7_final_holdout"
PROTO_PATH = ROOT / "ml" / "runs" / "week7_fashionclip_dev72" / "final_prototypes.npz"
RUN_DIR = ROOT / "ml" / "runs" / "week7_final_holdout"

MODEL_ID = "patrickjohncyh/fashion-clip"

STYLE_LABELS = [
    "Elegant", "Casual", "Romantic", "Minimalist", "Sporty",
    "Streetwear", "Boho", "Office", "Evening",
]
STYLE2ID = {s: i for i, s in enumerate(STYLE_LABELS)}
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)

def collect_holdout():
    rows = []
    counts = {}
    for style in STYLE_LABELS:
        folder = HOLDOUT_DIR / style
        if not folder.exists():
            raise SystemExit(f"Не найдена папка: {folder}")

        files = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in EXTS
        )
        counts[style] = len(files)

        for p in files:
            rows.append({
                "path": p,
                "file": str(p.relative_to(ROOT)).replace("\\", "/"),
                "style": style,
                "label": STYLE2ID[style],
            })

    print("=== FINAL HOLDOUT CHECK ===")
    for style in STYLE_LABELS:
        print(f"  {style:11s}: {counts[style]}")

    bad = {s: n for s, n in counts.items() if n != 3}
    if bad:
        raise SystemExit(
            "Для финального теста нужно ровно 3 фото на стиль. "
            f"Сейчас неверные количества: {bad}"
        )

    if len(rows) != 27:
        raise SystemExit(f"Ожидалось 27 фото, найдено {len(rows)}")

    return rows

def embed_image(model, processor, image, device):
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
        raise SystemExit(f"Не найден prototype-файл: {PROTO_PATH}")

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    rows = collect_holdout()

    proto = np.load(PROTO_PATH, allow_pickle=True)
    image_proto = normalize(proto["image_prototypes"].astype(np.float32))
    text_proto = normalize(proto["text_prototypes"].astype(np.float32))

    alpha_text = float(proto["alpha_text"][0])
    alpha_image = 1.0 - alpha_text

    print(f"\nLOCKED MODEL: text={alpha_text:.2f}, image={alpha_image:.2f}")
    print("Параметры на holdout НЕ подбираются.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Устройство:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    model.eval()

    xs = []
    ys = []
    files = []
    true_styles = []

    for i, row in enumerate(rows, start=1):
        image = Image.open(row["path"]).convert("RGB")
        xs.append(embed_image(model, processor, image, device))
        ys.append(row["label"])
        files.append(row["file"])
        true_styles.append(row["style"])
        print(f"[{i:02d}/27] {row['style']:11s} {row['path'].name}")

    x = normalize(np.stack(xs, axis=0).astype(np.float32))
    y = np.array(ys, dtype=np.int64)

    scores = alpha_text * (x @ text_proto.T) + alpha_image * (x @ image_proto.T)
    pred = scores.argmax(axis=1)

    acc = float(accuracy_score(y, pred))
    f1m = float(f1_score(y, pred, average="macro", zero_division=0))

    street_id = STYLE2ID["Streetwear"]
    street_fp = int(np.sum((pred == street_id) & (y != street_id)))

    p, r, f1, support = precision_recall_fscore_support(
        y,
        pred,
        labels=list(range(len(STYLE_LABELS))),
        zero_division=0,
    )

    per_class = pd.DataFrame({
        "style": STYLE_LABELS,
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": support,
    })
    per_class.to_csv(RUN_DIR / "per_class.csv", index=False)

    pred_rows = []
    for i, row in enumerate(rows):
        order = np.argsort(-scores[i])
        pred_rows.append({
            "file": row["file"],
            "true_style": row["style"],
            "pred_style": STYLE_LABELS[int(pred[i])],
            "correct": bool(pred[i] == y[i]),
            "score_top1": float(scores[i, order[0]]),
            "second_style": STYLE_LABELS[int(order[1])],
            "score_top2": float(scores[i, order[1]]),
            "margin": float(scores[i, order[0]] - scores[i, order[1]]),
        })

    pd.DataFrame(pred_rows).to_csv(
        RUN_DIR / "predictions.csv",
        index=False,
    )

    cm = confusion_matrix(
        y,
        pred,
        labels=list(range(len(STYLE_LABELS))),
    )
    pd.DataFrame(
        cm,
        index=STYLE_LABELS,
        columns=STYLE_LABELS,
    ).to_csv(RUN_DIR / "confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(STYLE_LABELS)))
    ax.set_yticks(range(len(STYLE_LABELS)))
    ax.set_xticklabels(STYLE_LABELS, rotation=45, ha="right")
    ax.set_yticklabels(STYLE_LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Week 7 Final Holdout Confusion Matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(RUN_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    summary = {
        "model": MODEL_ID,
        "method": "FashionCLIP locked text+real-image prototypes",
        "alpha_text": alpha_text,
        "alpha_image": alpha_image,
        "samples": int(len(y)),
        "accuracy": acc,
        "macro_f1": f1m,
        "streetwear_false_positives": street_fp,
    }
    (RUN_DIR / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== FINAL HOLDOUT RESULT ===")
    print(f"Samples={len(y)}")
    print(f"Locked mix: text={alpha_text:.2f} image={alpha_image:.2f}")
    print(f"Accuracy={acc:.3f}")
    print(f"Macro-F1={f1m:.3f}")
    print(f"Streetwear_FP={street_fp}")

    print("\nPer class:")
    for style, pp, rr, ff, ss in zip(STYLE_LABELS, p, r, f1, support):
        print(
            f"  {style:11s} | "
            f"P={pp:.3f} R={rr:.3f} F1={ff:.3f} n={int(ss)}"
        )

    print("\nSaved:")
    print(" ", RUN_DIR / "metrics_summary.json")
    print(" ", RUN_DIR / "per_class.csv")
    print(" ", RUN_DIR / "predictions.csv")
    print(" ", RUN_DIR / "confusion_matrix.csv")
    print(" ", RUN_DIR / "confusion_matrix.png")

if __name__ == "__main__":
    main()
