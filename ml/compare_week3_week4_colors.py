from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import ViTImageProcessor, ViTModel


ROOT = Path(__file__).resolve().parent.parent
ML_DIR = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "dataset" / "week4_color_images"
LABELS_CSV = ROOT / "dataset" / "week4_color_labels.csv"

WEEK3_MODEL = ML_DIR / "runs" / "week3" / "model.pt"
WEEK3_LABELS = ML_DIR / "runs" / "week3" / "label_maps.json"
WEEK4_MODEL = ML_DIR / "runs" / "week4" / "model.pt"
WEEK4_LABELS = ML_DIR / "runs" / "week4" / "label_maps.json"
OUT_DIR = ML_DIR / "runs" / "week4"

VIT_ID = "google/vit-base-patch16-224"
SEED = 42

OLD_COLORS = ["Black", "White", "Grey", "Beige", "Blue", "Brown", "Green", "Red", "Pink"]


class ColorDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor: ViTImageProcessor, labels: list[str]):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.color2id = {label: i for i, label in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(IMAGES_DIR / str(row["file"])).convert("RGB")
        x = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return x, str(row["color"])


class MultiHeadViT(nn.Module):
    def __init__(self, num_colors: int, num_styles: int):
        super().__init__()
        self.vit = ViTModel.from_pretrained(VIT_ID)
        hidden = self.vit.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.head_color = nn.Linear(hidden, num_colors)
        self.head_style = nn.Linear(hidden, num_styles)

    def forward(self, pixel_values):
        out = self.vit(pixel_values=pixel_values)
        pooled = self.dropout(out.pooler_output)
        return self.head_color(pooled)


def load_model(model_path: Path, labels_path: Path, device: torch.device):
    meta = json.loads(labels_path.read_text(encoding="utf-8"))
    colors = list(meta["color_labels"])
    styles = list(meta["style_labels"])

    model = MultiHeadViT(len(colors), len(styles))
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.to(device)
    model.eval()
    return model, colors


def evaluate(model, color_labels, loader, device):
    y_true: list[str] = []
    y_pred: list[str] = []

    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device)
            logits = model(x)
            ids = logits.argmax(dim=1).cpu().tolist()
            preds = [color_labels[i] for i in ids]

            y_true.extend(list(labels))
            y_pred.extend(preds)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                labels=OLD_COLORS,
                average="macro",
                zero_division=0,
            )
        ),
        "samples": len(y_true),
    }


def main():
    needed = [WEEK3_MODEL, WEEK3_LABELS, WEEK4_MODEL, WEEK4_LABELS, LABELS_CSV]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise SystemExit("Не найдены необходимые файлы:\n" + "\n".join(missing))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    df = pd.read_csv(LABELS_CSV, keep_default_na=False)

    # Воспроизводим тот же split, который использовался в week4 training.
    _, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["color"],
    )

    # Для корректного сравнения оставляем только 9 цветов,
    # которые существовали и в модели недели 3.
    common_val = val_df[val_df["color"].isin(OLD_COLORS)].copy()
    common_val = common_val.sort_values("file").reset_index(drop=True)

    processor = ViTImageProcessor.from_pretrained(VIT_ID)
    dataset = ColorDataset(common_val, processor, OLD_COLORS)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

    week3_model, week3_colors = load_model(WEEK3_MODEL, WEEK3_LABELS, device)
    week4_model, week4_colors = load_model(WEEK4_MODEL, WEEK4_LABELS, device)

    result3 = evaluate(week3_model, week3_colors, loader, device)
    result4 = evaluate(week4_model, week4_colors, loader, device)

    payload = {
        "benchmark": "same held-out week4 validation images; common 9 color classes only",
        "colors": OLD_COLORS,
        "week3": result3,
        "week4": result4,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "common_benchmark.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    labels = ["Accuracy", "F1"]
    before = [result3["accuracy"], result3["f1_macro"]]
    after = [result4["accuracy"], result4["f1_macro"]]
    x = [0, 1]
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar([v - width / 2 for v in x], before, width=width, label="Неделя 3")
    plt.bar([v + width / 2 for v in x], after, width=width, label="Неделя 4")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Значение")
    plt.title("Сравнение моделей на одной тестовой выборке")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "common_benchmark.png", dpi=180)
    plt.close()

    print("\nОбщая тестовая выборка:", len(common_val), "изображений")
    print(
        f"Week 3: Accuracy={result3['accuracy']:.3f}, "
        f"F1={result3['f1_macro']:.3f}"
    )
    print(
        f"Week 4: Accuracy={result4['accuracy']:.3f}, "
        f"F1={result4['f1_macro']:.3f}"
    )
    print(f"График: {OUT_DIR / 'common_benchmark.png'}")


if __name__ == "__main__":
    main()
