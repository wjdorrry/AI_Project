from __future__ import annotations

import json
import math
import os
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
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "week4_color_images"
LABELS_CSV = DATASET_DIR / "week4_color_labels.csv"

RUN_DIR = Path(__file__).resolve().parent / "runs" / "week4"
LATEST_DIR = Path(__file__).resolve().parent / "runs" / "latest"
WEEK3_DIR = Path(__file__).resolve().parent / "runs" / "week3"
WEEK3_MODEL = WEEK3_DIR / "model.pt"
WEEK3_LABELS = WEEK3_DIR / "label_maps.json"

VIT_ID = "google/vit-base-patch16-224"

OLD_COLOR_LABELS = ["Black", "White", "Grey", "Beige", "Blue", "Brown", "Green", "Red", "Pink"]
COLOR_LABELS = [
    "Black",
    "White",
    "Grey",
    "Beige",
    "Blue",
    "Brown",
    "Green",
    "Red",
    "Pink",
    "Burgundy",
    "Purple",
    "Yellow",
    "Orange",
    "Navy",
    "Cream",
]

STYLE_LABELS = [
    "Elegant",
    "Casual",
    "Romantic",
    "Minimalist",
    "Sporty",
    "Streetwear",
    "Boho",
    "Office",
    "Evening",
]

EPOCHS = int(os.getenv("SILLAGE_EPOCHS", "6"))
BATCH_SIZE = int(os.getenv("SILLAGE_BATCH_SIZE", "4"))
LEARNING_RATE = float(os.getenv("SILLAGE_LR", "5e-4"))
SEED = 42


class ColorDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor: ViTImageProcessor):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.color2id = {label: i for i, label in enumerate(COLOR_LABELS)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        image = Image.open(IMAGES_DIR / str(row["file"])).convert("RGB")
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


def load_week3_weights(model: MultiHeadViT) -> None:
    if not WEEK3_MODEL.exists():
        raise SystemExit(
            f"Не найдена лучшая модель 3-й недели: {WEEK3_MODEL}\n"
            "Скопируйте model.pt из ml/runs/week3 или сначала завершите обучение недели 3."
        )

    old_colors = OLD_COLOR_LABELS
    if WEEK3_LABELS.exists():
        payload = json.loads(WEEK3_LABELS.read_text(encoding="utf-8"))
        old_colors = payload.get("color_labels", old_colors)

    checkpoint = torch.load(WEEK3_MODEL, map_location="cpu", weights_only=False)
    source = checkpoint.get("model", checkpoint)
    target = model.state_dict()

    copied = 0
    for key, value in source.items():
        if key in {"head_color.weight", "head_color.bias"}:
            continue
        if key in target and target[key].shape == value.shape:
            target[key] = value
            copied += 1

    old_color2id = {label: i for i, label in enumerate(old_colors)}
    new_color2id = {label: i for i, label in enumerate(COLOR_LABELS)}

    if "head_color.weight" in source and "head_color.bias" in source:
        for label in OLD_COLOR_LABELS:
            if label not in old_color2id or label not in new_color2id:
                continue
            old_idx = old_color2id[label]
            new_idx = new_color2id[label]
            target["head_color.weight"][new_idx] = source["head_color.weight"][old_idx]
            target["head_color.bias"][new_idx] = source["head_color.bias"][old_idx]

    model.load_state_dict(target)
    print(f"Загружена модель 3-й недели. Перенесено параметров: {copied}")
    print("Старые 9 цветовых классов перенесены в новую голову из 15 классов.")


def freeze_style_and_backbone(model: MultiHeadViT) -> None:
    for p in model.vit.parameters():
        p.requires_grad = False
    for p in model.head_style.parameters():
        p.requires_grad = False

    for p in model.head_color.parameters():
        p.requires_grad = True

    print("ViT и style-голова заморожены; обучается только color-голова.")


def probability_errors(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float]:
    one_hot = torch.nn.functional.one_hot(targets, num_classes=len(COLOR_LABELS)).float()
    diff = probabilities - one_hot
    return float(diff.abs().mean().item()), float((diff ** 2).mean().item())


def evaluate(
    model: MultiHeadViT,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[dict[str, float], list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    batches = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    mae_total = 0.0
    mse_total = 0.0

    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].to(device)
            y = batch["y_color"].to(device)

            logits = model(x)["logits_color"]
            loss = loss_fn(logits, y)

            total_loss += float(loss.item())
            batches += 1

            probs = logits.softmax(dim=1)
            mae, mse = probability_errors(probs, y)
            mae_total += mae
            mse_total += mse

            y_true.extend(y.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    return (
        {
            "val_loss": total_loss / max(1, batches),
            "color_accuracy": float(accuracy_score(y_true, y_pred)),
            "color_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "color_mae": mae_total / max(1, batches),
            "color_mse": mse_total / max(1, batches),
        },
        y_true,
        y_pred,
    )


def save_dataset_distribution(df: pd.DataFrame) -> None:
    counts = df["color"].value_counts().reindex(COLOR_LABELS, fill_value=0)
    plt.figure(figsize=(12, 6))
    plt.bar(counts.index, counts.values)
    plt.title("Распределение изображений по цветам")
    plt.xlabel("Цвет")
    plt.ylabel("Количество изображений")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(RUN_DIR / "dataset_distribution.png", dpi=180)
    plt.close()


def save_history_plots(history: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], marker="o", label="Train loss")
    plt.plot(history["epoch"], history["val_loss"], marker="o", label="Validation loss")
    plt.title("Ошибка цветовой модели по эпохам")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RUN_DIR / "loss_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["color_accuracy"], marker="o", label="Color accuracy")
    plt.plot(history["epoch"], history["color_f1"], marker="o", label="Color F1")
    plt.title("Accuracy и F1 распознавания цвета")
    plt.xlabel("Эпоха")
    plt.ylabel("Значение")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RUN_DIR / "accuracy_f1_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["color_mae"], marker="o", label="Color MAE")
    plt.plot(history["epoch"], history["color_mse"], marker="o", label="Color MSE")
    plt.title("MAE и MSE распознавания цвета")
    plt.xlabel("Эпоха")
    plt.ylabel("Ошибка")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RUN_DIR / "mae_mse_curve.png", dpi=180)
    plt.close()


def save_confusion_matrix(y_true: list[int], y_pred: list[int]) -> None:
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
        title="Матрица ошибок по цветам",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(RUN_DIR / "color_confusion_matrix.png", dpi=180)
    plt.close(fig)


def main() -> None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    if not LABELS_CSV.exists():
        raise SystemExit(
            f"Не найден {LABELS_CSV}. Сначала запустите dataset/prepare_week4_color_dataset.py"
        )

    df = pd.read_csv(LABELS_CSV, keep_default_na=False)
    df["color"] = df["color"].astype(str).str.strip()

    bad = sorted(set(df["color"]) - set(COLOR_LABELS))
    if bad:
        raise SystemExit(f"Неизвестные цветовые классы: {bad}")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_DIR.mkdir(parents=True, exist_ok=True)

    save_dataset_distribution(df)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["color"],
    )

    processor = ViTImageProcessor.from_pretrained(VIT_ID)
    train_ds = ColorDataset(train_df, processor)
    val_ds = ColorDataset(val_df, processor)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = MultiHeadViT()
    load_week3_weights(model)
    freeze_style_and_backbone(model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )
    loss_fn = nn.CrossEntropyLoss()

    history: list[dict[str, float | int]] = []
    best_f1 = -math.inf
    best_true: list[int] = []
    best_pred: list[int] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        train_batches = 0

        bar = tqdm(train_loader, desc=f"train {epoch}/{EPOCHS}")
        for batch in bar:
            x = batch["pixel_values"].to(device)
            y = batch["y_color"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)["logits_color"]
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            total_train_loss += float(loss.item())
            train_batches += 1
            bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_train_loss / max(1, train_batches)
        metrics, y_true, y_pred = evaluate(model, val_loader, device, loss_fn)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics,
        }
        history.append(row)

        print(
            f"epoch {epoch}: "
            f"train_loss={train_loss:.4f} "
            f"val_loss={metrics['val_loss']:.4f} "
            f"color_acc={metrics['color_accuracy']:.3f} "
            f"color_f1={metrics['color_f1']:.3f} "
            f"color_mae={metrics['color_mae']:.4f} "
            f"color_mse={metrics['color_mse']:.4f}"
        )

        if metrics["color_f1"] > best_f1:
            best_f1 = metrics["color_f1"]
            best_true = list(y_true)
            best_pred = list(y_pred)

            checkpoint = {"model": model.state_dict()}
            torch.save(checkpoint, RUN_DIR / "model.pt")
            torch.save(checkpoint, LATEST_DIR / "model.pt")

            label_payload = {
                "color_labels": COLOR_LABELS,
                "style_labels": STYLE_LABELS,
                "vit_id": VIT_ID,
                "week": 4,
            }
            label_text = json.dumps(label_payload, ensure_ascii=False, indent=2)
            (RUN_DIR / "label_maps.json").write_text(label_text, encoding="utf-8")
            (LATEST_DIR / "label_maps.json").write_text(label_text, encoding="utf-8")

    history_df = pd.DataFrame(history)
    history_df.to_csv(RUN_DIR / "training_history.csv", index=False)

    save_history_plots(history_df)
    save_confusion_matrix(best_true, best_pred)

    best_row = history_df.loc[history_df["color_f1"].idxmax()].to_dict()
    summary = {
        "dataset_size": int(len(df)),
        "train_size": int(len(train_df)),
        "validation_size": int(len(val_df)),
        "color_labels": COLOR_LABELS,
        "style_labels_preserved": STYLE_LABELS,
        "best_epoch": int(best_row["epoch"]),
        "best_metrics": {
            k: float(v)
            for k, v in best_row.items()
            if k != "epoch" and not pd.isna(v)
        },
        "training_mode": "week3 transfer learning; ViT backbone and style head frozen; color head trained",
    }

    (RUN_DIR / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nОбучение завершено.")
    print(f"Лучшая эпоха: {summary['best_epoch']}")
    print(f"Color Accuracy: {best_row['color_accuracy']:.3f}")
    print(f"Color F1: {best_row['color_f1']:.3f}")
    print(f"Color MAE: {best_row['color_mae']:.4f}")
    print(f"Color MSE: {best_row['color_mse']:.4f}")
    print(f"Результаты и графики: {RUN_DIR}")


if __name__ == "__main__":
    main()
