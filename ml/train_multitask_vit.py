from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms
from transformers import ViTImageProcessor, ViTModel


ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_CSV = DATASET_DIR / "labels.csv"
LABELS_CLEAN_CSV = DATASET_DIR / "labels_clean.csv"

RUN_DIR = Path(__file__).resolve().parent / "runs" / "latest"

VIT_ID = "google/vit-base-patch16-224"

COLOR_LABELS = ["Black", "White", "Grey", "Beige", "Blue", "Brown", "Green", "Red", "Pink"]
STYLE_LABELS = ["Sportswear", "Evening", "Streetwear"]


@dataclass(frozen=True)
class LabelMaps:
    color2id: dict[str, int]
    id2color: dict[int, str]
    style2id: dict[str, int]
    id2style: dict[int, str]


class OutfitDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor: ViTImageProcessor, label_maps: LabelMaps, augment: bool):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.label_maps = label_maps
        self.augment = augment
        self._aug = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.14, hue=0.03),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        path = IMAGES_DIR / str(row["file"])
        img = Image.open(path).convert("RGB")
        if self.augment:
            img = self._aug(img)
        enc = self.processor(images=img, return_tensors="pt")
        pixel_values = enc["pixel_values"][0]  # [3,224,224]

        color = str(row["color"]).strip()
        style = str(row["style"]).strip()
        y_color = self.label_maps.color2id[color]
        y_style = self.label_maps.style2id[style]

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
        pooled = out.pooler_output  # [B,H]
        pooled = self.dropout(pooled)
        return {
            "logits_color": self.head_color(pooled),
            "logits_style": self.head_style(pooled),
        }


def load_labels() -> pd.DataFrame:
    csv_path = LABELS_CLEAN_CSV if LABELS_CLEAN_CSV.exists() else LABELS_CSV
    if not csv_path.exists():
        raise SystemExit(f"labels file not found at: {csv_path}")
    df = pd.read_csv(csv_path, comment="#")
    needed = {"file", "color", "style"}
    if not needed.issubset(set(df.columns)):
        raise SystemExit(f"labels.csv must have columns: {sorted(needed)}")
    df["file"] = df["file"].astype(str)
    df["color"] = df["color"].astype(str).str.strip()
    df["style"] = df["style"].astype(str).str.strip()

    has_file = df["file"].apply(lambda f: (IMAGES_DIR / f).exists())
    dropped = int((~has_file).sum())
    if dropped:
        print(f"WARNING: {dropped} rows in CSV have no matching file in {IMAGES_DIR.name}/ — skipped.")
    df = df[has_file].reset_index(drop=True)
    if len(df) == 0:
        raise SystemExit(f"No rows left: add images to {IMAGES_DIR} matching the 'file' column in labels CSV.")

    bad_color = sorted(set(df["color"]) - set(COLOR_LABELS))
    bad_style = sorted(set(df["style"]) - set(STYLE_LABELS))
    if bad_color:
        raise SystemExit(f"Unknown color labels in CSV: {bad_color}. Allowed: {COLOR_LABELS}")
    if bad_style:
        raise SystemExit(f"Unknown style labels in CSV: {bad_style}. Allowed: {STYLE_LABELS}")

    # sanity: must have at least 2 classes in each task to learn anything meaningful
    if df["style"].nunique() < 2:
        print("WARNING: only one style class in dataset. Add Evening and Streetwear to learn style.")
    if df["color"].nunique() < 2:
        print("WARNING: only one color class in dataset. Add more colors to learn color.")

    return df


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_labels()

    label_maps = LabelMaps(
        color2id={c: i for i, c in enumerate(COLOR_LABELS)},
        id2color={i: c for i, c in enumerate(COLOR_LABELS)},
        style2id={s: i for i, s in enumerate(STYLE_LABELS)},
        id2style={i: s for i, s in enumerate(STYLE_LABELS)},
    )

    processor = ViTImageProcessor.from_pretrained(VIT_ID)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["style"] if df["style"].nunique() > 1 else None)

    train_ds = OutfitDataset(train_df, processor, label_maps, augment=True)
    val_ds = OutfitDataset(val_df, processor, label_maps, augment=False)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)

    model = MultiHeadViT(VIT_ID, num_colors=len(COLOR_LABELS), num_styles=len(STYLE_LABELS)).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    ce = nn.CrossEntropyLoss()

    epochs = 5
    lam = 1.0  # вес style-loss

    best_val = 10**9
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"train {epoch}/{epochs}")
        total_loss = 0.0
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            y_color = batch["y_color"].to(device)
            y_style = batch["y_style"].to(device)

            out = model(pixel_values=pixel_values)
            loss_color = ce(out["logits_color"], y_color)
            loss_style = ce(out["logits_style"], y_style)
            loss = loss_color + lam * loss_style

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            pbar.set_postfix(loss=float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                y_color = batch["y_color"].to(device)
                y_style = batch["y_style"].to(device)
                out = model(pixel_values=pixel_values)
                loss = ce(out["logits_color"], y_color) + lam * ce(out["logits_style"], y_style)
                val_loss += float(loss.item())
            val_loss /= max(1, len(val_loader))

        print(f"epoch {epoch}: train_loss={total_loss / max(1, len(train_loader)):.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            RUN_DIR.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict()}, RUN_DIR / "model.pt")
            (RUN_DIR / "label_maps.json").write_text(
                json.dumps(
                    {
                        "color_labels": COLOR_LABELS,
                        "style_labels": STYLE_LABELS,
                        "vit_id": VIT_ID,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    print(f"Saved best model to: {RUN_DIR}")


if __name__ == "__main__":
    main()

