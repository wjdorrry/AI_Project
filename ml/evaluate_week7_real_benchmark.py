from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import ViTImageProcessor, ViTModel

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = ROOT / "dataset" / "week7_real_benchmark"
RUN_DIR = ROOT / "ml" / "runs" / "week7_style_vit"
MODEL_PATH = RUN_DIR / "model.pt"
OUT_DIR = RUN_DIR / "real_benchmark"
VIT_ID = "google/vit-base-patch16-224"

STYLE_LABELS = [
    "Elegant", "Casual", "Romantic", "Minimalist", "Sporty",
    "Streetwear", "Boho", "Office", "Evening",
]
COLOR_LABELS = [
    "Black", "White", "Grey", "Beige", "Blue", "Brown", "Green", "Red", "Pink",
    "Burgundy", "Purple", "Yellow", "Orange", "Navy", "Cream",
]
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


class MultiHeadViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained(VIT_ID)
        hidden = self.vit.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head_color = nn.Linear(hidden, len(COLOR_LABELS))
        self.head_style = nn.Linear(hidden, len(STYLE_LABELS))

    def forward(self, pixel_values):
        out = self.vit(pixel_values=pixel_values)
        pooled = self.dropout(out.pooler_output)
        return {
            "logits_color": self.head_color(pooled),
            "logits_style": self.head_style(pooled),
        }


class BenchmarkDataset(Dataset):
    def __init__(self, rows, processor):
        self.rows = rows
        self.processor = processor
        self.style2id = {s: i for i, s in enumerate(STYLE_LABELS)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(row["path"]).convert("RGB")
        enc = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": enc["pixel_values"][0],
            "y_style": torch.tensor(self.style2id[row["style"]], dtype=torch.long),
            "file": row["file"],
            "style": row["style"],
        }


def collect_rows():
    rows = []
    for style in STYLE_LABELS:
        folder = BENCHMARK_DIR / style
        if not folder.exists():
            continue
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.suffix.lower() in EXTS:
                rows.append({
                    "path": p,
                    "file": str(p.relative_to(ROOT)).replace("\\", "/"),
                    "style": style,
                })
    return rows


def save_confusion(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(STYLE_LABELS))))
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(STYLE_LABELS)),
        yticks=range(len(STYLE_LABELS)),
        xticklabels=STYLE_LABELS,
        yticklabels=STYLE_LABELS,
        xlabel="Предсказанный стиль",
        ylabel="Правильный стиль",
        title="Week 7 — real photo benchmark",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=8,
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"Не найдена модель: {MODEL_PATH}")

    rows = collect_rows()
    if not rows:
        raise SystemExit(
            "Нет изображений. Разложите фото по папкам "
            "dataset/week7_real_benchmark/<STYLE>/"
        )

    counts = pd.Series([r["style"] for r in rows]).value_counts()
    missing = [s for s in STYLE_LABELS if counts.get(s, 0) == 0]

    print("Найдено изображений:", len(rows))
    print("По стилям:", counts.to_dict())
    if missing:
        print("ВНИМАНИЕ: нет фото для:", ", ".join(missing))
    if any(counts.get(s, 0) < 3 for s in STYLE_LABELS):
        print("ВНИМАНИЕ: для честного мини-benchmark желательно минимум 3 фото на стиль.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Устройство:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    processor = ViTImageProcessor.from_pretrained(VIT_ID)
    ds = BenchmarkDataset(rows, processor)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)

    model = MultiHeadViT()
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()

    results = []
    y_true = []
    y_pred = []

    with torch.no_grad():
        offset = 0
        for batch in loader:
            x = batch["pixel_values"].to(device)
            y = batch["y_style"].to(device)
            logits = model(x)["logits_style"]
            probs = logits.softmax(dim=1)
            top2 = probs.topk(k=2, dim=1)
            pred = top2.indices[:, 0]

            for i in range(len(pred)):
                true_id = int(y[i].item())
                pred_id = int(pred[i].item())
                conf = float(top2.values[i, 0].item())
                second_id = int(top2.indices[i, 1].item())
                second_conf = float(top2.values[i, 1].item())
                results.append({
                    "file": batch["file"][i],
                    "true_style": STYLE_LABELS[true_id],
                    "pred_style": STYLE_LABELS[pred_id],
                    "correct": pred_id == true_id,
                    "confidence": conf,
                    "second_style": STYLE_LABELS[second_id],
                    "second_confidence": second_conf,
                    "margin": conf - second_conf,
                })
                y_true.append(true_id)
                y_pred.append(pred_id)
            offset += len(pred)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT_DIR / "predictions.csv", index=False)

    accuracy = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
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
    per_class.to_csv(OUT_DIR / "per_class.csv", index=False)
    save_confusion(y_true, y_pred, OUT_DIR / "confusion_matrix.png")

    sid = STYLE_LABELS.index("Streetwear")
    streetwear_fp = sum(1 for y, p_ in zip(y_true, y_pred) if p_ == sid and y != sid)

    summary = {
        "samples": len(rows),
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "streetwear_false_positives": streetwear_fp,
        "mean_confidence": float(result_df["confidence"].mean()),
        "mean_margin": float(result_df["margin"].mean()),
        "counts": {s: int(counts.get(s, 0)) for s in STYLE_LABELS},
    }
    (OUT_DIR / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== REAL BENCHMARK ===")
    print(f"Samples={len(rows)}")
    print(f"Accuracy={accuracy:.3f}")
    print(f"Macro-F1={f1_macro:.3f}")
    print(f"Streetwear_FP={streetwear_fp}")
    print(f"Mean confidence={summary['mean_confidence']:.3f}")
    print(f"Mean margin={summary['mean_margin']:.3f}")
    print("Результаты:", OUT_DIR)


if __name__ == "__main__":
    main()
