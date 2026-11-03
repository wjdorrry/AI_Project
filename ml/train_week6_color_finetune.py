from __future__ import annotations

import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "week6_color_images"
LABELS_CSV = DATASET_DIR / "week6_color_labels.csv"

WEEK3_IMAGES = DATASET_DIR / "week3_images"
WEEK3_LABELS_CSV = DATASET_DIR / "week3_labels.csv"

RUN_DIR = Path(__file__).resolve().parent / "runs" / "week6_color_finetune"
BASELINE_DIR = Path(__file__).resolve().parent / "runs" / "week4"
BASELINE_MODEL = BASELINE_DIR / "model.pt"
BASELINE_LABELS = BASELINE_DIR / "label_maps.json"

VIT_ID = "google/vit-base-patch16-224"

COLOR_LABELS = [
    "Black", "White", "Grey", "Beige", "Blue", "Brown", "Green", "Red", "Pink",
    "Burgundy", "Purple", "Yellow", "Orange", "Navy", "Cream",
]
STYLE_LABELS = [
    "Elegant", "Casual", "Romantic", "Minimalist", "Sporty",
    "Streetwear", "Boho", "Office", "Evening",
]

EPOCHS = int(os.getenv("SILLAGE_EPOCHS", "6"))
BATCH_SIZE = int(os.getenv("SILLAGE_BATCH_SIZE", "2"))
GRAD_ACCUM = int(os.getenv("SILLAGE_GRAD_ACCUM", "2"))
HEAD_LR = float(os.getenv("SILLAGE_HEAD_LR", "5e-5"))
BACKBONE_LR = float(os.getenv("SILLAGE_BACKBONE_LR", "2e-6"))
SEED = 52


class ColorDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor: ViTImageProcessor, augment: bool):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.augment = augment
        self.color2id = {label: i for i, label in enumerate(COLOR_LABELS)}
        self.aug = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.90, 1.0), ratio=(0.94, 1.06)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.03),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        image = Image.open(IMAGES_DIR / str(row["file"])).convert("RGB")
        if self.augment:
            image = self.aug(image)
        enc = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": enc["pixel_values"][0],
            "y_color": torch.tensor(self.color2id[str(row["color"])], dtype=torch.long),
        }


class StyleDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor: ViTImageProcessor):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.style2id = {label: i for i, label in enumerate(STYLE_LABELS)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        image = Image.open(WEEK3_IMAGES / str(row["file"])).convert("RGB")
        enc = self.processor(images=image, return_tensors="pt")
        return {
            "pixel_values": enc["pixel_values"][0],
            "y_style": torch.tensor(self.style2id[str(row["style"])], dtype=torch.long),
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


def load_baseline(model: MultiHeadViT) -> None:
    if not BASELINE_MODEL.exists():
        raise SystemExit(f"Не найдена модель недели 4: {BASELINE_MODEL}")

    checkpoint = torch.load(BASELINE_MODEL, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    model.load_state_dict(state, strict=True)
    print("Загружена лучшая модель недели 4 (15 цветов).")


def configure_finetuning(model: MultiHeadViT) -> int:
    for p in model.parameters():
        p.requires_grad = False

    for p in model.head_color.parameters():
        p.requires_grad = True

    layer_ids: list[int] = []
    for name, _ in model.named_parameters():
        match = re.search(r"vit\.(?:layers|encoder\.layer)\.(\d+)\.", name)
        if match:
            layer_ids.append(int(match.group(1)))

    if not layer_ids:
        raise RuntimeError("Не удалось определить слои ViT для разморозки.")

    last_layer = max(layer_ids)
    prefixes = (
        f"vit.layers.{last_layer}.",
        f"vit.encoder.layer.{last_layer}.",
    )

    for name, p in model.named_parameters():
        if name.startswith(prefixes):
            p.requires_grad = True

    print(f"Разморожен последний блок ViT: {last_layer}.")
    print("Style-голова заморожена. Color-голова обучается.")
    return last_layer


def probability_errors(probabilities: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    one_hot = torch.nn.functional.one_hot(targets, num_classes=len(COLOR_LABELS)).float()
    diff = probabilities - one_hot
    return float(diff.abs().mean().item()), float((diff ** 2).mean().item())


def evaluate_color(
    model: MultiHeadViT,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[dict[str, float], list[int], list[int]]:
    model.eval()
    losses = 0.0
    batches = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    mae_sum = 0.0
    mse_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].to(device)
            y = batch["y_color"].to(device)
            logits = model(x)["logits_color"]
            loss = loss_fn(logits, y)
            probs = logits.softmax(dim=1)
            mae, mse = probability_errors(probs, y)

            losses += float(loss.item())
            batches += 1
            mae_sum += mae
            mse_sum += mse
            y_true.extend(y.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    return (
        {
            "loss": losses / max(1, batches),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "mae": mae_sum / max(1, batches),
            "mse": mse_sum / max(1, batches),
        },
        y_true,
        y_pred,
    )


def evaluate_style(model: MultiHeadViT, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].to(device)
            y = batch["y_style"].to(device)
            logits = model(x)["logits_style"]
            y_true.extend(y.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def save_confusion(y_true: list[int], y_pred: list[int], path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(COLOR_LABELS))))
    fig, ax = plt.subplots(figsize=(12, 11))
    im = ax.imshow(cm)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(COLOR_LABELS)),
        yticks=range(len(COLOR_LABELS)),
        xticklabels=COLOR_LABELS,
        yticklabels=COLOR_LABELS,
        xlabel="Предсказанный цвет",
        ylabel="Правильный цвет",
        title="Матрица ошибок улучшенной цветовой модели",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_history(history: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], marker="o", label="Train loss")
    plt.plot(history["epoch"], history["val_loss"], marker="o", label="Validation loss")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.title("Дообучение распознавания цвета")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RUN_DIR / "loss_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["val_accuracy"], marker="o", label="Accuracy")
    plt.plot(history["epoch"], history["val_f1"], marker="o", label="F1 macro")
    plt.ylim(0, 1)
    plt.xlabel("Эпоха")
    plt.ylabel("Значение")
    plt.title("Accuracy и F1 после разморозки последнего блока ViT")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RUN_DIR / "accuracy_f1_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["val_mae"], marker="o", label="MAE")
    plt.plot(history["epoch"], history["val_mse"], marker="o", label="MSE")
    plt.xlabel("Эпоха")
    plt.ylabel("Ошибка")
    plt.title("MAE и MSE улучшенной цветовой модели")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RUN_DIR / "mae_mse_curve.png", dpi=180)
    plt.close()


def save_comparison(baseline: dict[str, float], improved: dict[str, float]) -> None:
    labels = ["Accuracy", "F1"]
    before = [baseline["accuracy"], baseline["f1_macro"]]
    after = [improved["accuracy"], improved["f1_macro"]]
    x = list(range(len(labels)))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar([v - width / 2 for v in x], before, width=width, label="Week 4 baseline")
    plt.bar([v + width / 2 for v in x], after, width=width, label="Fine-tuned")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Значение")
    plt.title("До и после на одной тестовой выборке")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RUN_DIR / "before_after_same_test.png", dpi=180)
    plt.close()


def main() -> None:
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not LABELS_CSV.exists():
        raise SystemExit(
            f"Не найден {LABELS_CSV}. Сначала запустите dataset/prepare_week6_color_dataset.py"
        )
    if not BASELINE_MODEL.exists():
        raise SystemExit(f"Не найдена модель недели 4: {BASELINE_MODEL}")

    df = pd.read_csv(LABELS_CSV, keep_default_na=False)
    df["color"] = df["color"].astype(str).str.strip()
    bad = sorted(set(df["color"]) - set(COLOR_LABELS))
    if bad:
        raise SystemExit(f"Неизвестные цветовые классы: {bad}")

    train_val_df, test_df = train_test_split(
        df, test_size=0.15, random_state=SEED, stratify=df["color"]
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.1764705882,
        random_state=SEED,
        stratify=train_val_df["color"],
    )

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(RUN_DIR / "train_split.csv", index=False)
    val_df.to_csv(RUN_DIR / "validation_split.csv", index=False)
    test_df.to_csv(RUN_DIR / "test_split.csv", index=False)

    print(f"Датасет: {len(df)}")
    print(f"Train: {len(train_df)} | Validation: {len(val_df)} | Test: {len(test_df)}")

    processor = ViTImageProcessor.from_pretrained(VIT_ID)
    train_loader = DataLoader(
        ColorDataset(train_df, processor, augment=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        ColorDataset(val_df, processor, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        ColorDataset(test_df, processor, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    baseline = MultiHeadViT()
    load_baseline(baseline)
    baseline = baseline.to(device)

    loss_fn = nn.CrossEntropyLoss()
    baseline_test, _, _ = evaluate_color(baseline, test_loader, device, loss_fn)
    print(
        "Baseline Week 4 на новом test: "
        f"Accuracy={baseline_test['accuracy']:.3f}, F1={baseline_test['f1_macro']:.3f}"
    )

    model = MultiHeadViT()
    load_baseline(model)
    last_layer = configure_finetuning(model)
    model = model.to(device)

    head_params = [p for p in model.head_color.parameters() if p.requires_grad]
    backbone_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("head_color.")
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": HEAD_LR},
            {"params": backbone_params, "lr": BACKBONE_LR},
        ],
        weight_decay=0.01,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history: list[dict[str, float | int]] = []
    best_f1 = -math.inf
    best_true: list[int] = []
    best_pred: list[int] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        batches = 0

        bar = tqdm(train_loader, desc=f"fine-tune {epoch}/{EPOCHS}")
        for step, batch in enumerate(bar, start=1):
            x = batch["pixel_values"].to(device)
            y = batch["y_color"].to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)["logits_color"]
                loss = loss_fn(logits, y) / GRAD_ACCUM

            scaler.scale(loss).backward()

            if step % GRAD_ACCUM == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item()) * GRAD_ACCUM
            batches += 1
            bar.set_postfix(loss=f"{float(loss.item()) * GRAD_ACCUM:.4f}")

        train_loss = total_loss / max(1, batches)
        val_metrics, y_true, y_pred = evaluate_color(model, val_loader, device, loss_fn)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1_macro"],
            "val_mae": val_metrics["mae"],
            "val_mse": val_metrics["mse"],
        }
        history.append(row)

        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"color_acc={val_metrics['accuracy']:.3f} "
            f"color_f1={val_metrics['f1_macro']:.3f} "
            f"color_mae={val_metrics['mae']:.4f} "
            f"color_mse={val_metrics['mse']:.4f}"
        )

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_true = list(y_true)
            best_pred = list(y_pred)
            torch.save({"model": model.state_dict()}, RUN_DIR / "model.pt")

    history_df = pd.DataFrame(history)
    history_df.to_csv(RUN_DIR / "training_history.csv", index=False)
    save_history(history_df)

    best_model = MultiHeadViT()
    checkpoint = torch.load(RUN_DIR / "model.pt", map_location="cpu", weights_only=False)
    best_model.load_state_dict(checkpoint["model"], strict=True)
    best_model = best_model.to(device)

    improved_test, test_true, test_pred = evaluate_color(best_model, test_loader, device, loss_fn)
    save_confusion(test_true, test_pred, RUN_DIR / "color_confusion_matrix_test.png")
    save_comparison(baseline_test, improved_test)

    style_comparison = None
    if WEEK3_LABELS_CSV.exists() and WEEK3_IMAGES.exists():
        style_df = pd.read_csv(WEEK3_LABELS_CSV, keep_default_na=False)
        style_df["style"] = style_df["style"].astype(str).str.strip()
        _, style_val = train_test_split(
            style_df, test_size=0.2, random_state=42, stratify=style_df["style"]
        )
        style_loader = DataLoader(
            StyleDataset(style_val, processor),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )
        baseline_style = evaluate_style(baseline, style_loader, device)
        improved_style = evaluate_style(best_model, style_loader, device)
        style_comparison = {
            "baseline_week4": baseline_style,
            "improved": improved_style,
        }
        print(
            "Style check: "
            f"Week4 F1={baseline_style['f1_macro']:.3f} -> "
            f"Improved F1={improved_style['f1_macro']:.3f}"
        )

    best_row = history_df.loc[history_df["val_f1"].idxmax()].to_dict()
    summary = {
        "dataset_size": int(len(df)),
        "train_size": int(len(train_df)),
        "validation_size": int(len(val_df)),
        "test_size": int(len(test_df)),
        "color_labels": COLOR_LABELS,
        "style_labels": STYLE_LABELS,
        "starting_model": "week4/model.pt",
        "unfrozen_last_vit_block": int(last_layer),
        "head_lr": HEAD_LR,
        "backbone_lr": BACKBONE_LR,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": GRAD_ACCUM,
        "best_epoch": int(best_row["epoch"]),
        "best_validation": {
            "accuracy": float(best_row["val_accuracy"]),
            "f1_macro": float(best_row["val_f1"]),
            "mae": float(best_row["val_mae"]),
            "mse": float(best_row["val_mse"]),
            "loss": float(best_row["val_loss"]),
        },
        "same_test_comparison": {
            "baseline_week4": baseline_test,
            "improved": improved_test,
        },
        "style_preservation_check": style_comparison,
    }

    (RUN_DIR / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RUN_DIR / "label_maps.json").write_text(
        json.dumps(
            {
                "color_labels": COLOR_LABELS,
                "style_labels": STYLE_LABELS,
                "vit_id": VIT_ID,
                "week": 6,
                "stage": "color_finetune",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nДообучение завершено.")
    print(f"Лучшая эпоха: {summary['best_epoch']}")
    print(
        "Одна и та же test-выборка: "
        f"Week4 Accuracy={baseline_test['accuracy']:.3f}, F1={baseline_test['f1_macro']:.3f} | "
        f"Improved Accuracy={improved_test['accuracy']:.3f}, F1={improved_test['f1_macro']:.3f}"
    )
    print(f"Результаты: {RUN_DIR}")
    print("ВАЖНО: ml/runs/latest пока не изменён. Новая модель сначала должна пройти сравнение.")


if __name__ == "__main__":
    main()
