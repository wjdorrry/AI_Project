from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "dataset" / "week7_style_balanced" / "manifest.csv"

RUN_DIR = ROOT / "ml" / "runs" / "week7_style_vit"
BASE_MODEL = ROOT / "ml" / "runs" / "week6_color_7500" / "model.pt"

COLOR_IMAGES = ROOT / "dataset" / "week6_color_7500_images"
COLOR_LABELS_CSV = ROOT / "dataset" / "week6_color_7500_labels.csv"

VIT_ID = "google/vit-base-patch16-224"

STYLE_LABELS = [
    "Elegant", "Casual", "Romantic", "Minimalist", "Sporty",
    "Streetwear", "Boho", "Office", "Evening",
]
COLOR_LABELS = [
    "Black", "White", "Grey", "Beige", "Blue", "Brown", "Green", "Red", "Pink",
    "Burgundy", "Purple", "Yellow", "Orange", "Navy", "Cream",
]

HEAD_ONLY_EPOCHS = int(os.getenv("SILLAGE_STAGE1_EPOCHS", "3"))
FINETUNE_EPOCHS = int(os.getenv("SILLAGE_STAGE2_EPOCHS", "5"))
BATCH_SIZE = int(os.getenv("SILLAGE_BATCH_SIZE", "2"))
GRAD_ACCUM = int(os.getenv("SILLAGE_GRAD_ACCUM", "2"))
HEAD_LR_STAGE1 = float(os.getenv("SILLAGE_STYLE_HEAD_LR1", "3e-4"))
HEAD_LR_STAGE2 = float(os.getenv("SILLAGE_STYLE_HEAD_LR2", "8e-5"))
BACKBONE_LR = float(os.getenv("SILLAGE_STYLE_BACKBONE_LR", "2e-6"))
WEIGHT_DECAY = float(os.getenv("SILLAGE_WEIGHT_DECAY", "0.01"))
LABEL_SMOOTHING = float(os.getenv("SILLAGE_LABEL_SMOOTHING", "0.05"))
SEED = 20261117
COLOR_SPLIT_SEED = 52


class StyleDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor: ViTImageProcessor, augment: bool):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.augment = augment
        self.style2id = {label: i for i, label in enumerate(STYLE_LABELS)}
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.88, 1.0), ratio=(0.92, 1.08)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=4),
            transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        image = Image.open(ROOT / str(row["file"])).convert("RGB")
        if self.augment:
            image = self.aug(image)
        enc = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": enc["pixel_values"][0],
            "y_style": torch.tensor(self.style2id[str(row["style"])], dtype=torch.long),
        }


class ColorDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor: ViTImageProcessor):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.color2id = {label: i for i, label in enumerate(COLOR_LABELS)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        image = Image.open(COLOR_IMAGES / str(row["file"])).convert("RGB")
        enc = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": enc["pixel_values"][0],
            "y_color": torch.tensor(self.color2id[str(row["color"])], dtype=torch.long),
        }


class MultiHeadViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = ViTModel.from_pretrained(VIT_ID)
        hidden = self.vit.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head_color = nn.Linear(hidden, len(COLOR_LABELS))
        self.head_style = nn.Linear(hidden, len(STYLE_LABELS))

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.vit(pixel_values=pixel_values)
        pooled = self.dropout(out.pooler_output)
        return {
            "logits_color": self.head_color(pooled),
            "logits_style": self.head_style(pooled),
        }


def set_seed() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def load_week6(model: MultiHeadViT) -> None:
    if not BASE_MODEL.exists():
        raise SystemExit(f"Не найдена модель недели 6: {BASE_MODEL}")
    checkpoint = torch.load(BASE_MODEL, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    print("Загружена модель недели 6 (7500 цветов).")


def find_last_vit_layer(model: MultiHeadViT) -> int:
    ids = []
    for name, _ in model.named_parameters():
        m = re.search(r"vit\.(?:layers|encoder\.layer)\.(\d+)\.", name)
        if m:
            ids.append(int(m.group(1)))
    if not ids:
        raise RuntimeError("Не удалось найти блоки ViT.")
    return max(ids)


def freeze_for_stage1(model: MultiHeadViT) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head_style.parameters():
        p.requires_grad = True
    print("Этап 1: обучается только style-head. Цвет и ViT заморожены.")


def configure_stage2(model: MultiHeadViT) -> int:
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head_style.parameters():
        p.requires_grad = True

    last = find_last_vit_layer(model)
    prefixes = (
        f"vit.layers.{last}.",
        f"vit.encoder.layer.{last}.",
    )
    for name, p in model.named_parameters():
        if name.startswith(prefixes):
            p.requires_grad = True

    print(f"Этап 2: style-head + последний блок ViT ({last}).")
    print("Color-head остаётся замороженной.")
    return last


def build_balanced_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    counts = train_df["style"].value_counts().to_dict()
    weights = [1.0 / counts[str(style)] for style in train_df["style"]]
    generator = torch.Generator()
    generator.manual_seed(SEED)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_df),
        replacement=True,
        generator=generator,
    )


def evaluate_style(model, loader, device, loss_fn):
    model.eval()
    losses = 0.0
    batches = 0
    y_true, y_pred = [], []
    confidences, margins = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].to(device)
            y = batch["y_style"].to(device)
            logits = model(x)["logits_style"]
            loss = loss_fn(logits, y)
            probs = logits.softmax(dim=1)
            top2 = probs.topk(k=2, dim=1)

            losses += float(loss.item())
            batches += 1
            y_true.extend(y.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            confidences.extend(top2.values[:, 0].cpu().tolist())
            margins.extend((top2.values[:, 0] - top2.values[:, 1]).cpu().tolist())

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {
        "loss": losses / max(1, batches),
        "accuracy": acc,
        "f1_macro": f1,
        "mean_top1_confidence": float(sum(confidences) / max(1, len(confidences))),
        "mean_top1_top2_margin": float(sum(margins) / max(1, len(margins))),
    }, y_true, y_pred


def per_class_table(y_true, y_pred) -> pd.DataFrame:
    p, r, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(STYLE_LABELS))),
        zero_division=0,
    )
    return pd.DataFrame({
        "style": STYLE_LABELS,
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": support,
    })


def streetwear_diagnostics(y_true, y_pred) -> dict[str, int | float]:
    sid = STYLE_LABELS.index("Streetwear")
    pred_count = sum(1 for p in y_pred if p == sid)
    true_count = sum(1 for y in y_true if y == sid)
    false_positive = sum(1 for y, p in zip(y_true, y_pred) if p == sid and y != sid)
    false_negative = sum(1 for y, p in zip(y_true, y_pred) if y == sid and p != sid)
    total = len(y_true)
    return {
        "predicted_streetwear": pred_count,
        "true_streetwear": true_count,
        "streetwear_false_positives": false_positive,
        "streetwear_false_negatives": false_negative,
        "predicted_streetwear_share": pred_count / max(1, total),
    }


def save_confusion(y_true, y_pred, path: Path, title: str) -> None:
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
        title=title,
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


def save_history(history: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    for stage in sorted(history["stage"].unique()):
        part = history[history["stage"] == stage]
        plt.plot(part["global_epoch"], part["val_f1"], marker="o", label=f"{stage} F1")
    plt.xlabel("Эпоха")
    plt.ylabel("Macro-F1")
    plt.ylim(0, 1)
    plt.title("Week 7: качество распознавания стиля")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RUN_DIR / "style_f1_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["global_epoch"], history["train_loss"], marker="o", label="Train loss")
    plt.plot(history["global_epoch"], history["val_loss"], marker="o", label="Validation loss")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.title("Week 7: обучение стиля")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RUN_DIR / "style_loss_curve.png", dpi=180)
    plt.close()


def train_stage(
    model,
    train_loader,
    val_loader,
    device,
    loss_fn,
    optimizer,
    scaler,
    stage_name,
    epochs,
    start_epoch,
    best_f1,
):
    history = []
    for local_epoch in range(1, epochs + 1):
        global_epoch = start_epoch + local_epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        batches = 0

        bar = tqdm(train_loader, desc=f"{stage_name} {local_epoch}/{epochs}")
        for step, batch in enumerate(bar, start=1):
            x = batch["pixel_values"].to(device)
            y = batch["y_style"].to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)["logits_style"]
                loss = loss_fn(logits, y) / GRAD_ACCUM

            scaler.scale(loss).backward()
            if step % GRAD_ACCUM == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            full_loss = float(loss.item()) * GRAD_ACCUM
            total_loss += full_loss
            batches += 1
            bar.set_postfix(loss=f"{full_loss:.4f}")

        train_loss = total_loss / max(1, batches)
        val_metrics, val_true, val_pred = evaluate_style(model, val_loader, device, loss_fn)
        diag = streetwear_diagnostics(val_true, val_pred)

        row = {
            "stage": stage_name,
            "local_epoch": local_epoch,
            "global_epoch": global_epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1_macro"],
            "val_confidence": val_metrics["mean_top1_confidence"],
            "val_margin": val_metrics["mean_top1_top2_margin"],
            "val_streetwear_false_positives": diag["streetwear_false_positives"],
        }
        history.append(row)

        print(
            f"epoch {global_epoch}: train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"acc={val_metrics['accuracy']:.3f} "
            f"f1={val_metrics['f1_macro']:.3f} "
            f"Streetwear_FP={diag['streetwear_false_positives']}"
        )

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "stage": stage_name,
                    "epoch": global_epoch,
                    "val_f1": best_f1,
                },
                RUN_DIR / "model.pt",
            )
            per_class_table(val_true, val_pred).to_csv(
                RUN_DIR / "best_validation_per_class.csv", index=False
            )
            save_confusion(
                val_true,
                val_pred,
                RUN_DIR / "best_validation_confusion_matrix.png",
                "Week 7 — validation confusion matrix",
            )

    return history, best_f1


def evaluate_color_preservation(model, processor, device):
    if not COLOR_LABELS_CSV.exists() or not COLOR_IMAGES.exists():
        return None

    df = pd.read_csv(COLOR_LABELS_CSV, keep_default_na=False)
    df["color"] = df["color"].astype(str).str.strip()
    _, test_df = train_test_split(
        df,
        test_size=0.15,
        random_state=COLOR_SPLIT_SEED,
        stratify=df["color"],
    )

    loader = DataLoader(
        ColorDataset(test_df, processor),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="color preservation"):
            x = batch["pixel_values"].to(device)
            y = batch["y_color"].to(device)
            logits = model(x)["logits_color"]
            y_true.extend(y.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "samples": len(y_true),
    }


def main() -> None:
    set_seed()
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Устройство:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    if not MANIFEST.exists():
        raise SystemExit(f"Не найден manifest: {MANIFEST}")
    if not BASE_MODEL.exists():
        raise SystemExit(f"Не найдена модель недели 6: {BASE_MODEL}")

    df = pd.read_csv(MANIFEST, keep_default_na=False)
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()

    print(f"Train={len(train_df)} | Validation={len(val_df)} | Test={len(test_df)}")
    print("Train counts:", train_df["style"].value_counts().to_dict())

    processor = ViTImageProcessor.from_pretrained(VIT_ID)

    sampler = build_balanced_sampler(train_df)
    train_loader = DataLoader(
        StyleDataset(train_df, processor, augment=True),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        StyleDataset(val_df, processor, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        StyleDataset(test_df, processor, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    baseline = MultiHeadViT()
    load_week6(baseline)
    baseline = baseline.to(device)

    baseline_test, base_true, base_pred = evaluate_style(
        baseline, test_loader, device, loss_fn
    )
    baseline_diag = streetwear_diagnostics(base_true, base_pred)
    per_class_table(base_true, base_pred).to_csv(
        RUN_DIR / "baseline_test_per_class.csv", index=False
    )
    save_confusion(
        base_true,
        base_pred,
        RUN_DIR / "baseline_test_confusion_matrix.png",
        "Week 6 baseline — test confusion matrix",
    )
    print(
        "Baseline Week 6: "
        f"Accuracy={baseline_test['accuracy']:.3f}, "
        f"F1={baseline_test['f1_macro']:.3f}, "
        f"Streetwear_FP={baseline_diag['streetwear_false_positives']}"
    )

    model = MultiHeadViT()
    load_week6(model)
    model = model.to(device)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    best_f1 = -math.inf
    all_history = []

    freeze_for_stage1(model)
    opt1 = torch.optim.AdamW(
        [p for p in model.head_style.parameters() if p.requires_grad],
        lr=HEAD_LR_STAGE1,
        weight_decay=WEIGHT_DECAY,
    )
    hist1, best_f1 = train_stage(
        model, train_loader, val_loader, device, loss_fn,
        opt1, scaler, "stage1_head", HEAD_ONLY_EPOCHS, 0, best_f1
    )
    all_history.extend(hist1)

    # Начинаем второй этап с лучшей модели первого этапа.
    best_stage1 = torch.load(RUN_DIR / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_stage1["model"], strict=True)

    last_layer = configure_stage2(model)
    style_params = [p for p in model.head_style.parameters() if p.requires_grad]
    backbone_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("head_style.")
    ]
    opt2 = torch.optim.AdamW(
        [
            {"params": style_params, "lr": HEAD_LR_STAGE2},
            {"params": backbone_params, "lr": BACKBONE_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    hist2, best_f1 = train_stage(
        model, train_loader, val_loader, device, loss_fn,
        opt2, scaler, "stage2_last_block", FINETUNE_EPOCHS,
        HEAD_ONLY_EPOCHS, best_f1
    )
    all_history.extend(hist2)

    history_df = pd.DataFrame(all_history)
    history_df.to_csv(RUN_DIR / "training_history.csv", index=False)
    save_history(history_df)

    best_checkpoint = torch.load(
        RUN_DIR / "model.pt", map_location="cpu", weights_only=False
    )
    best_model = MultiHeadViT()
    best_model.load_state_dict(best_checkpoint["model"], strict=True)
    best_model = best_model.to(device)

    final_test, test_true, test_pred = evaluate_style(
        best_model, test_loader, device, loss_fn
    )
    final_diag = streetwear_diagnostics(test_true, test_pred)
    final_per_class = per_class_table(test_true, test_pred)
    final_per_class.to_csv(RUN_DIR / "final_test_per_class.csv", index=False)
    save_confusion(
        test_true,
        test_pred,
        RUN_DIR / "final_test_confusion_matrix.png",
        "Week 7 improved ViT — test confusion matrix",
    )

    color_before = evaluate_color_preservation(baseline, processor, device)
    color_after = evaluate_color_preservation(best_model, processor, device)

    summary = {
        "dataset": {
            "originals": int(len(df)),
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
            "train_counts": train_df["style"].value_counts().to_dict(),
        },
        "style_labels": STYLE_LABELS,
        "starting_model": str(BASE_MODEL),
        "training": {
            "stage1_epochs": HEAD_ONLY_EPOCHS,
            "stage2_epochs": FINETUNE_EPOCHS,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRAD_ACCUM,
            "stage1_head_lr": HEAD_LR_STAGE1,
            "stage2_head_lr": HEAD_LR_STAGE2,
            "stage2_backbone_lr": BACKBONE_LR,
            "label_smoothing": LABEL_SMOOTHING,
            "unfrozen_last_vit_block": int(last_layer),
            "best_checkpoint_stage": best_checkpoint.get("stage"),
            "best_checkpoint_epoch": int(best_checkpoint.get("epoch", -1)),
            "best_validation_f1": float(best_checkpoint.get("val_f1", 0.0)),
        },
        "same_test_comparison": {
            "week6_baseline": baseline_test,
            "week7_improved": final_test,
        },
        "streetwear_diagnostics": {
            "week6_baseline": baseline_diag,
            "week7_improved": final_diag,
        },
        "color_preservation": {
            "week6_baseline": color_before,
            "week7_improved": color_after,
        },
    }

    (RUN_DIR / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RUN_DIR / "label_maps.json").write_text(
        json.dumps({
            "color_labels": COLOR_LABELS,
            "style_labels": STYLE_LABELS,
            "vit_id": VIT_ID,
            "week": 7,
            "stage": "style_quality_vit",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== WEEK 7 RESULT ===")
    print(
        f"Week 6 baseline: Accuracy={baseline_test['accuracy']:.3f} "
        f"F1={baseline_test['f1_macro']:.3f} "
        f"Streetwear_FP={baseline_diag['streetwear_false_positives']}"
    )
    print(
        f"Week 7 improved: Accuracy={final_test['accuracy']:.3f} "
        f"F1={final_test['f1_macro']:.3f} "
        f"Streetwear_FP={final_diag['streetwear_false_positives']}"
    )
    if color_before and color_after:
        print(
            f"Color preservation: F1 {color_before['f1_macro']:.3f} "
            f"-> {color_after['f1_macro']:.3f}"
        )
    print("Best checkpoint:", RUN_DIR / "model.pt")
    print("ВАЖНО: runs/latest автоматически не менялся.")


if __name__ == "__main__":
    main()
