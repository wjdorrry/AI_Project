from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
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
IMAGES_DIR = DATASET_DIR / "week3_images"
LABELS_CSV = DATASET_DIR / "week3_labels.csv"

RUN_DIR = Path(__file__).resolve().parent / "runs" / "week3"
LATEST_DIR = Path(__file__).resolve().parent / "runs" / "latest"

VIT_ID = "google/vit-base-patch16-224"

COLOR_LABELS = ["Black", "White", "Grey", "Beige", "Blue", "Brown", "Green", "Red", "Pink"]
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
BATCH_SIZE = int(os.getenv("SILLAGE_BATCH_SIZE", "2"))
LEARNING_RATE = float(os.getenv("SILLAGE_LR", "2e-5"))
SEED = 42


@dataclass(frozen=True)
class LabelMaps:
    color2id: dict[str, int]
    id2color: dict[int, str]
    style2id: dict[str, int]
    id2style: dict[int, str]


class Week3Dataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        processor: ViTImageProcessor,
        maps: LabelMaps,
        augment: bool,
    ):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.maps = maps
        self.augment = augment
        self.aug = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.88, 1.0), ratio=(0.92, 1.08)),
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        path = IMAGES_DIR / str(row["file"])
        image = Image.open(path).convert("RGB")
        if self.augment:
            image = self.aug(image)

        enc = self.processor(images=image, return_tensors="pt")
        pixel_values = enc["pixel_values"][0]

        style = str(row["style"]).strip()
        color = str(row.get("color", "")).strip()

        # Внешние изображения имеют style, но не имеют нашей 9-классовой color-разметки.
        # -100 используется как ignore_index для color loss.
        y_color = self.maps.color2id[color] if color in self.maps.color2id else -100
        y_style = self.maps.style2id[style]

        return {
            "pixel_values": pixel_values,
            "y_color": torch.tensor(y_color, dtype=torch.long),
            "y_style": torch.tensor(y_style, dtype=torch.long),
        }


class MultiHeadViT(nn.Module):
    def __init__(self, vit_id: str, num_colors: int, num_styles: int):
        super().__init__()
        self.vit = ViTModel.from_pretrained(vit_id)
        hidden = self.vit.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head_color = nn.Linear(hidden, num_colors)
        self.head_style = nn.Linear(hidden, num_styles)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.vit(pixel_values=pixel_values)
        pooled = self.dropout(out.pooler_output)
        return {
            "logits_color": self.head_color(pooled),
            "logits_style": self.head_style(pooled),
        }


def load_labels() -> pd.DataFrame:
    if not LABELS_CSV.exists():
        raise SystemExit(
            f"Не найден {LABELS_CSV}. Сначала запустите dataset/prepare_week3_dataset.py"
        )

    df = pd.read_csv(LABELS_CSV, keep_default_na=False)
    needed = {"file", "color", "style"}
    if not needed.issubset(df.columns):
        raise SystemExit(f"В week3_labels.csv нужны столбцы: {sorted(needed)}")

    df["file"] = df["file"].astype(str)
    df["color"] = df["color"].astype(str).str.strip()
    df["style"] = df["style"].astype(str).str.strip()

    exists = df["file"].apply(lambda x: (IMAGES_DIR / x).exists())
    if not bool(exists.all()):
        missing = int((~exists).sum())
        raise SystemExit(f"Не найдено изображений из CSV: {missing}")

    bad_style = sorted(set(df["style"]) - set(STYLE_LABELS))
    if bad_style:
        raise SystemExit(f"Неизвестные style-классы: {bad_style}")

    unknown_colors = sorted({x for x in df["color"] if x and x not in COLOR_LABELS})
    if unknown_colors:
        raise SystemExit(f"Неизвестные color-классы: {unknown_colors}")

    counts = Counter(df["style"])
    missing_styles = [s for s in STYLE_LABELS if counts.get(s, 0) == 0]
    if missing_styles:
        raise SystemExit(f"В датасете нет примеров классов: {missing_styles}")

    return df


def probability_errors(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> tuple[float, float]:
    one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).float()
    diff = probabilities - one_hot
    mae = float(diff.abs().mean().item())
    mse = float((diff ** 2).mean().item())
    return mae, mse


def evaluate(
    model: MultiHeadViT,
    loader: DataLoader,
    device: torch.device,
    ce_style: nn.Module,
    ce_color: nn.Module,
) -> tuple[dict[str, float], list[int], list[int]]:
    model.eval()

    total_loss = 0.0
    batches = 0

    style_true: list[int] = []
    style_pred: list[int] = []
    color_true: list[int] = []
    color_pred: list[int] = []

    style_mae_sum = 0.0
    style_mse_sum = 0.0
    style_metric_batches = 0

    color_mae_sum = 0.0
    color_mse_sum = 0.0
    color_metric_batches = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["pixel_values"].to(device)
            y_style = batch["y_style"].to(device)
            y_color = batch["y_color"].to(device)

            out = model(x)
            logits_style = out["logits_style"]
            logits_color = out["logits_color"]

            loss_style = ce_style(logits_style, y_style)

            valid_color = y_color >= 0
            if bool(valid_color.any()):
                loss_color = ce_color(logits_color[valid_color], y_color[valid_color])
            else:
                loss_color = torch.zeros((), device=device)

            loss = loss_style + loss_color
            total_loss += float(loss.item())
            batches += 1

            style_prob = logits_style.softmax(dim=1)
            s_mae, s_mse = probability_errors(style_prob, y_style, len(STYLE_LABELS))
            style_mae_sum += s_mae
            style_mse_sum += s_mse
            style_metric_batches += 1

            style_true.extend(y_style.cpu().tolist())
            style_pred.extend(logits_style.argmax(dim=1).cpu().tolist())

            if bool(valid_color.any()):
                valid_logits = logits_color[valid_color]
                valid_targets = y_color[valid_color]
                color_prob = valid_logits.softmax(dim=1)
                c_mae, c_mse = probability_errors(color_prob, valid_targets, len(COLOR_LABELS))
                color_mae_sum += c_mae
                color_mse_sum += c_mse
                color_metric_batches += 1
                color_true.extend(valid_targets.cpu().tolist())
                color_pred.extend(valid_logits.argmax(dim=1).cpu().tolist())

    metrics = {
        "val_loss": total_loss / max(1, batches),
        "style_accuracy": float(accuracy_score(style_true, style_pred)),
        "style_f1": float(f1_score(style_true, style_pred, average="macro", zero_division=0)),
        "style_mae": style_mae_sum / max(1, style_metric_batches),
        "style_mse": style_mse_sum / max(1, style_metric_batches),
        "color_accuracy": float(accuracy_score(color_true, color_pred)) if color_true else math.nan,
        "color_f1": float(f1_score(color_true, color_pred, average="macro", zero_division=0)) if color_true else math.nan,
        "color_mae": color_mae_sum / max(1, color_metric_batches) if color_true else math.nan,
        "color_mse": color_mse_sum / max(1, color_metric_batches) if color_true else math.nan,
    }
    return metrics, style_true, style_pred


def save_dataset_distribution(df: pd.DataFrame) -> None:
    counts = df["style"].value_counts().reindex(STYLE_LABELS, fill_value=0)
    plt.figure(figsize=(10, 5))
    plt.bar(counts.index, counts.values)
    plt.title("Распределение изображений по стилям")
    plt.xlabel("Стиль")
    plt.ylabel("Количество изображений")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(RUN_DIR / "dataset_distribution.png", dpi=180)
    plt.close()


def save_history_plots(history: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], marker="o", label="Train loss")
    plt.plot(history["epoch"], history["val_loss"], marker="o", label="Validation loss")
    plt.title("Ошибка модели по эпохам")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RUN_DIR / "loss_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(history["epoch"], history["style_accuracy"], marker="o", label="Style accuracy")
    plt.plot(history["epoch"], history["style_f1"], marker="o", label="Style F1")
    plt.plot(history["epoch"], history["color_accuracy"], marker="o", label="Color accuracy")
    plt.plot(history["epoch"], history["color_f1"], marker="o", label="Color F1")
    plt.title("Accuracy и F1 по эпохам")
    plt.xlabel("Эпоха")
    plt.ylabel("Значение")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(RUN_DIR / "accuracy_f1_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(history["epoch"], history["style_mae"], marker="o", label="Style MAE")
    plt.plot(history["epoch"], history["style_mse"], marker="o", label="Style MSE")
    plt.plot(history["epoch"], history["color_mae"], marker="o", label="Color MAE")
    plt.plot(history["epoch"], history["color_mse"], marker="o", label="Color MSE")
    plt.title("MAE и MSE по эпохам")
    plt.xlabel("Эпоха")
    plt.ylabel("Ошибка")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RUN_DIR / "mae_mse_curve.png", dpi=180)
    plt.close()


def save_confusion_matrix(y_true: list[int], y_pred: list[int]) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(STYLE_LABELS))))
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=range(len(STYLE_LABELS)),
        yticks=range(len(STYLE_LABELS)),
        xticklabels=STYLE_LABELS,
        yticklabels=STYLE_LABELS,
        xlabel="Предсказанный класс",
        ylabel="Правильный класс",
        title="Матрица ошибок по стилям",
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
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(RUN_DIR / "style_confusion_matrix.png", dpi=180)
    plt.close(fig)


def main() -> None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    df = load_labels()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_DIR.mkdir(parents=True, exist_ok=True)

    save_dataset_distribution(df)

    maps = LabelMaps(
        color2id={c: i for i, c in enumerate(COLOR_LABELS)},
        id2color={i: c for i, c in enumerate(COLOR_LABELS)},
        style2id={s: i for i, s in enumerate(STYLE_LABELS)},
        id2style={i: s for i, s in enumerate(STYLE_LABELS)},
    )

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["style"],
    )

    processor = ViTImageProcessor.from_pretrained(VIT_ID)
    train_ds = Week3Dataset(train_df, processor, maps, augment=True)
    val_ds = Week3Dataset(val_df, processor, maps, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MultiHeadViT(VIT_ID, len(COLOR_LABELS), len(STYLE_LABELS)).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    ce_style = nn.CrossEntropyLoss()
    ce_color = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history: list[dict[str, float | int]] = []
    best_style_f1 = -1.0
    best_style_true: list[int] = []
    best_style_pred: list[int] = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        train_batches = 0

        bar = tqdm(train_loader, desc=f"train {epoch}/{EPOCHS}")
        for batch in bar:
            x = batch["pixel_values"].to(device)
            y_style = batch["y_style"].to(device)
            y_color = batch["y_color"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(x)
                loss_style = ce_style(out["logits_style"], y_style)

                valid_color = y_color >= 0
                if bool(valid_color.any()):
                    loss_color = ce_color(out["logits_color"][valid_color], y_color[valid_color])
                else:
                    loss_color = torch.zeros((), device=device)

                loss = loss_style + loss_color

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += float(loss.item())
            train_batches += 1
            bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = total_train_loss / max(1, train_batches)

        metrics, style_true, style_pred = evaluate(
            model,
            val_loader,
            device,
            ce_style,
            ce_color,
        )

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
            f"style_acc={metrics['style_accuracy']:.3f} "
            f"style_f1={metrics['style_f1']:.3f} "
            f"style_mae={metrics['style_mae']:.4f} "
            f"style_mse={metrics['style_mse']:.4f} "
            f"color_acc={metrics['color_accuracy']:.3f} "
            f"color_f1={metrics['color_f1']:.3f}"
        )

        if metrics["style_f1"] > best_style_f1:
            best_style_f1 = metrics["style_f1"]
            best_style_true = list(style_true)
            best_style_pred = list(style_pred)

            checkpoint = {"model": model.state_dict()}
            torch.save(checkpoint, RUN_DIR / "model.pt")
            torch.save(checkpoint, LATEST_DIR / "model.pt")

            label_payload = {
                "color_labels": COLOR_LABELS,
                "style_labels": STYLE_LABELS,
                "vit_id": VIT_ID,
                "week": 3,
            }
            text = json.dumps(label_payload, ensure_ascii=False, indent=2)
            (RUN_DIR / "label_maps.json").write_text(text, encoding="utf-8")
            (LATEST_DIR / "label_maps.json").write_text(text, encoding="utf-8")

    history_df = pd.DataFrame(history)
    history_df.to_csv(RUN_DIR / "training_history.csv", index=False)

    save_history_plots(history_df)
    save_confusion_matrix(best_style_true, best_style_pred)

    best_row = history_df.loc[history_df["style_f1"].idxmax()].to_dict()
    summary = {
        "dataset_size": int(len(df)),
        "train_size": int(len(train_df)),
        "validation_size": int(len(val_df)),
        "style_labels": STYLE_LABELS,
        "color_labels": COLOR_LABELS,
        "best_epoch": int(best_row["epoch"]),
        "best_metrics": {
            k: (None if pd.isna(v) else float(v))
            for k, v in best_row.items()
            if k != "epoch"
        },
    }
    (RUN_DIR / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nОбучение завершено.")
    print(f"Лучшая эпоха: {summary['best_epoch']}")
    print(f"Style Accuracy: {best_row['style_accuracy']:.3f}")
    print(f"Style F1: {best_row['style_f1']:.3f}")
    print(f"Style MAE: {best_row['style_mae']:.4f}")
    print(f"Style MSE: {best_row['style_mse']:.4f}")
    print(f"Результаты и графики: {RUN_DIR}")


if __name__ == "__main__":
    main()
