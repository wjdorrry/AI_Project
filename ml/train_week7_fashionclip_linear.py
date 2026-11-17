from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "dataset" / "week7_style_balanced" / "manifest.csv"
REAL_DIR = ROOT / "dataset" / "week7_real_benchmark"
RUN_DIR = ROOT / "ml" / "runs" / "week7_fashionclip_linear"
MODEL_ID = "patrickjohncyh/fashion-clip"

STYLE_LABELS = [
    "Elegant", "Casual", "Romantic", "Minimalist", "Sporty",
    "Streetwear", "Boho", "Office", "Evening",
]
STYLE2ID = {s: i for i, s in enumerate(STYLE_LABELS)}
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

BATCH_SIZE = int(os.getenv("SILLAGE_FCLIP_BATCH_SIZE", "4"))
NUM_WORKERS = 0
C_VALUES = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


class ImagePathDataset(Dataset):
    def __init__(self, rows: list[dict], processor):
        self.rows = rows
        self.processor = processor

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(row["path"]).convert("RGB")
        enc = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": enc["pixel_values"][0],
            "label": torch.tensor(STYLE2ID[row["style"]], dtype=torch.long),
            "file": row["file"],
        }


def manifest_rows(df: pd.DataFrame, split: str) -> list[dict]:
    rows = []
    part = df[df["split"] == split]
    for _, row in part.iterrows():
        rel = str(row["file"]).replace("\\", "/")
        rows.append({
            "path": ROOT / rel,
            "file": rel,
            "style": str(row["style"]),
        })
    return rows


def real_rows() -> list[dict]:
    rows = []
    for style in STYLE_LABELS:
        folder = REAL_DIR / style
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


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def extract_embeddings(name, rows, processor, model, device):
    cache = RUN_DIR / f"{name}_embeddings.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        print(f"{name}: использую сохранённые embeddings ({len(data['y'])})")
        return data["x"], data["y"], data["files"].tolist()

    loader = DataLoader(
        ImagePathDataset(rows, processor),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    xs, ys, files = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"embeddings {name}"):
            pixels = batch["pixel_values"].to(device)
            feats = model.get_image_features(pixel_values=pixels)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            xs.append(feats.float().cpu().numpy())
            ys.append(batch["label"].cpu().numpy())
            files.extend(list(batch["file"]))

    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.int64)
    x = l2_normalize(x)
    np.savez_compressed(cache, x=x, y=y, files=np.array(files, dtype=object))
    return x, y, files


def evaluate(name, clf, x, y, files):
    pred = clf.predict(x)
    probs = clf.predict_proba(x)
    order = np.argsort(-probs, axis=1)
    top1 = order[:, 0]
    top2 = order[:, 1]
    conf = probs[np.arange(len(y)), top1]
    second = probs[np.arange(len(y)), top2]
    margin = conf - second

    acc = float(accuracy_score(y, pred))
    f1m = float(f1_score(y, pred, average="macro", zero_division=0))

    p, r, f1, support = precision_recall_fscore_support(
        y, pred, labels=list(range(len(STYLE_LABELS))), zero_division=0
    )
    pd.DataFrame({
        "style": STYLE_LABELS,
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": support,
    }).to_csv(RUN_DIR / f"{name}_per_class.csv", index=False)

    rows = []
    for i in range(len(y)):
        rows.append({
            "file": files[i],
            "true_style": STYLE_LABELS[int(y[i])],
            "pred_style": STYLE_LABELS[int(pred[i])],
            "correct": bool(pred[i] == y[i]),
            "confidence": float(conf[i]),
            "second_style": STYLE_LABELS[int(top2[i])],
            "second_confidence": float(second[i]),
            "margin": float(margin[i]),
        })
    pd.DataFrame(rows).to_csv(RUN_DIR / f"{name}_predictions.csv", index=False)

    cm = confusion_matrix(y, pred, labels=list(range(len(STYLE_LABELS))))
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
        title=f"FashionCLIP linear — {name}",
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
    fig.savefig(RUN_DIR / f"{name}_confusion_matrix.png", dpi=180)
    plt.close(fig)

    sid = STYLE2ID["Streetwear"]
    streetwear_fp = int(sum(1 for yt, yp in zip(y, pred) if yp == sid and yt != sid))

    return {
        "samples": int(len(y)),
        "accuracy": acc,
        "f1_macro": f1m,
        "streetwear_false_positives": streetwear_fp,
        "mean_confidence": float(np.mean(conf)),
        "mean_margin": float(np.mean(margin)),
    }


def main():
    if not MANIFEST.exists():
        raise SystemExit(f"Не найден manifest: {MANIFEST}")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MANIFEST, keep_default_na=False)

    train_rows = manifest_rows(df, "train")
    val_rows = manifest_rows(df, "validation")
    test_rows = manifest_rows(df, "test")
    dev_rows = real_rows()

    if not dev_rows:
        raise SystemExit("Не найден real benchmark: dataset/week7_real_benchmark/<STYLE>/")

    print(
        f"Train={len(train_rows)} | Validation={len(val_rows)} | "
        f"Internal test={len(test_rows)} | Real dev={len(dev_rows)}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Устройство:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("FashionCLIP:", MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    model.eval()

    x_train, y_train, train_files = extract_embeddings(
        "train", train_rows, processor, model, device
    )
    x_val, y_val, val_files = extract_embeddings(
        "validation", val_rows, processor, model, device
    )
    x_test, y_test, test_files = extract_embeddings(
        "internal_test", test_rows, processor, model, device
    )
    x_dev, y_dev, dev_files = extract_embeddings(
        "real_dev", dev_rows, processor, model, device
    )

    print("\nПодбор C по validation Macro-F1:")
    search_rows = []
    best_c = None
    best_f1 = -1.0

    for c in C_VALUES:
        clf = LogisticRegression(
            C=c,
            max_iter=4000,
            class_weight="balanced",
            solver="lbfgs",
        )
        clf.fit(x_train, y_train)
        pred = clf.predict(x_val)
        acc = float(accuracy_score(y_val, pred))
        f1m = float(f1_score(y_val, pred, average="macro", zero_division=0))
        search_rows.append({"C": c, "val_accuracy": acc, "val_f1_macro": f1m})
        print(f"C={c:<5} val_acc={acc:.3f} val_f1={f1m:.3f}")
        if f1m > best_f1:
            best_f1 = f1m
            best_c = c

    pd.DataFrame(search_rows).to_csv(RUN_DIR / "c_search.csv", index=False)
    print(f"Лучшее C: {best_c} | val F1={best_f1:.3f}")

    # После выбора гиперпараметра обучаем финальную линейную голову на train+validation.
    x_fit = np.concatenate([x_train, x_val], axis=0)
    y_fit = np.concatenate([y_train, y_val], axis=0)

    final_clf = LogisticRegression(
        C=best_c,
        max_iter=4000,
        class_weight="balanced",
        solver="lbfgs",
    )
    final_clf.fit(x_fit, y_fit)

    with (RUN_DIR / "classifier.pkl").open("wb") as f:
        pickle.dump(final_clf, f)

    internal = evaluate("internal_test", final_clf, x_test, y_test, test_files)
    real_dev = evaluate("real_dev", final_clf, x_dev, y_dev, dev_files)

    summary = {
        "encoder": MODEL_ID,
        "classifier": "LogisticRegression",
        "best_C": best_c,
        "best_validation_f1": best_f1,
        "internal_test": internal,
        "real_dev": real_dev,
    }
    (RUN_DIR / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== FASHIONCLIP + LINEAR CLASSIFIER ===")
    print(
        f"Internal test: Accuracy={internal['accuracy']:.3f} "
        f"F1={internal['f1_macro']:.3f} "
        f"Streetwear_FP={internal['streetwear_false_positives']}"
    )
    print(
        f"Real dev:      Accuracy={real_dev['accuracy']:.3f} "
        f"F1={real_dev['f1_macro']:.3f} "
        f"Streetwear_FP={real_dev['streetwear_false_positives']}"
    )
    print(
        f"Real dev confidence={real_dev['mean_confidence']:.3f} "
        f"margin={real_dev['mean_margin']:.3f}"
    )
    print("Результаты:", RUN_DIR)
    print(
        "Важно: текущие 27 реальных фото считаем development-набором. "
        "Финальную модель позже проверим на новом отдельном real holdout."
    )


if __name__ == "__main__":
    main()
