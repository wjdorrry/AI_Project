from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import AutoProcessor, CLIPModel

ROOT = Path(__file__).resolve().parent.parent
SFS_MANIFEST = ROOT / "dataset" / "sfs_combined_real" / "manifest.csv"
DEV_DIRS = [
    ROOT / "dataset" / "week7_real_benchmark",
    ROOT / "dataset" / "week7_real_holdout",
]
RUN_DIR = ROOT / "ml" / "runs" / "week7_sfs_prototypes"
MODEL_ID = "patrickjohncyh/fashion-clip"

STYLE_LABELS = [
    "Elegant", "Casual", "Romantic", "Minimalist", "Sporty",
    "Streetwear", "Boho", "Office", "Evening",
]
STYLE2ID = {s: i for i, s in enumerate(STYLE_LABELS)}
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

PROMPTS = {
    "Elegant": [
        "an elegant polished sophisticated fashion outfit",
        "a refined elegant outfit with polished styling",
        "sophisticated elegant clothing with a refined appearance",
    ],
    "Casual": [
        "a casual everyday fashion outfit",
        "comfortable everyday casual clothing",
        "a relaxed casual look for daily wear",
    ],
    "Romantic": [
        "a romantic soft delicate fashion outfit",
        "a romantic outfit with soft feminine details",
        "romantic clothing with delicate graceful styling",
    ],
    "Minimalist": [
        "a minimalist clean simple fashion outfit with clean lines",
        "a monochrome or neutral minimalist outfit with simple structured silhouettes and little decoration",
        "understated minimalist clothing with clean tailoring, no busy prints and minimal details",
    ],
    "Sporty": [
        "a sporty athletic fashion outfit",
        "athletic activewear and sporty clothing",
        "a sports inspired outfit with functional athletic styling",
    ],
    "Streetwear": [
        "a streetwear urban fashion outfit",
        "urban streetwear with relaxed modern styling",
        "a street style outfit with sneakers and contemporary urban clothing",
    ],
    "Boho": [
        "a bohemian boho fashion outfit",
        "a boho outfit with flowing earthy artistic details",
        "bohemian clothing with relaxed layered natural styling",
    ],
    "Office": [
        "a professional office business fashion outfit",
        "business office clothing with tailored professional styling",
        "a professional work outfit with blazer or tailored pieces",
    ],
    "Evening": [
        "an evening formal fashion outfit",
        "a formal evening look for a party or event",
        "dressy evening clothing with formal event styling",
    ],
}

ALPHAS = [0.0, 0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 1.0]

def normalize(x):
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)

def read_manifest():
    df = pd.read_csv(SFS_MANIFEST, keep_default_na=False)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "path": ROOT / str(r["file"]),
            "file": str(r["file"]),
            "style": str(r["style"]),
            "label": STYLE2ID[str(r["style"])],
        })
    return rows

def collect_dev():
    rows = []
    for base in DEV_DIRS:
        for style in STYLE_LABELS:
            folder = base / style
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

def image_embedding(model, processor, image, device):
    enc = processor(images=image, return_tensors="pt")
    pixels = enc["pixel_values"].to(device)
    with torch.no_grad():
        out = model.vision_model(pixel_values=pixels)
        pooled = out.pooler_output
        feat = model.visual_projection(pooled)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0].float().cpu().numpy()

def text_prototypes(model, processor, device):
    protos = []
    with torch.no_grad():
        for style in STYLE_LABELS:
            enc = processor(text=PROMPTS[style], return_tensors="pt", padding=True)
            out = model.text_model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            )
            feats = model.text_projection(out.pooler_output)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            proto = feats.mean(dim=0)
            proto = proto / proto.norm()
            protos.append(proto.float().cpu().numpy())
    return normalize(np.stack(protos, axis=0))

def embed_rows(name, rows, model, processor, device):
    cache = RUN_DIR / f"{name}_embeddings.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        return data["x"], data["y"], data["files"].tolist()

    xs, ys, files = [], [], []
    for i, row in enumerate(rows, start=1):
        image = Image.open(row["path"]).convert("RGB")
        xs.append(image_embedding(model, processor, image, device))
        ys.append(row["label"])
        files.append(row["file"])
        if i % 50 == 0 or i == len(rows):
            print(f"{name}: {i}/{len(rows)}")

    x = normalize(np.stack(xs, axis=0).astype(np.float32))
    y = np.array(ys, dtype=np.int64)
    np.savez_compressed(cache, x=x, y=y, files=np.array(files, dtype=object))
    return x, y, files

def build_class_prototypes(x, y):
    protos = []
    for class_id in range(len(STYLE_LABELS)):
        part = x[y == class_id]
        if len(part) == 0:
            raise RuntimeError(f"Нет примеров для {STYLE_LABELS[class_id]}")
        proto = normalize(part.mean(axis=0, keepdims=True))[0]
        protos.append(proto)
    return np.stack(protos, axis=0)

def main():
    if not SFS_MANIFEST.exists():
        raise SystemExit(f"Не найден: {SFS_MANIFEST}")

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    sfs_rows = read_manifest()
    dev_rows = collect_dev()

    print("SFS real train:", len(sfs_rows))
    print("Real dev:", len(dev_rows))
    print("SFS counts:", pd.Series([r["style"] for r in sfs_rows]).value_counts().to_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Устройство:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    model.eval()

    x_sfs, y_sfs, sfs_files = embed_rows("sfs", sfs_rows, model, processor, device)
    x_dev, y_dev, dev_files = embed_rows("dev72", dev_rows, model, processor, device)

    image_proto = build_class_prototypes(x_sfs, y_sfs)
    text_proto = text_prototypes(model, processor, device)

    search = []
    best = None

    for alpha in ALPHAS:
        scores = alpha * (x_dev @ text_proto.T) + (1.0 - alpha) * (x_dev @ image_proto.T)
        pred = scores.argmax(axis=1)

        acc = float(accuracy_score(y_dev, pred))
        f1m = float(f1_score(y_dev, pred, average="macro", zero_division=0))
        sid = STYLE2ID["Streetwear"]
        fp = int(sum(1 for yt, yp in zip(y_dev, pred) if yp == sid and yt != sid))

        row = {
            "alpha_text": alpha,
            "alpha_sfs": 1.0 - alpha,
            "accuracy": acc,
            "f1_macro": f1m,
            "streetwear_false_positives": fp,
        }
        search.append(row)
        print(
            f"alpha_text={alpha:.2f} | acc={acc:.3f} "
            f"F1={f1m:.3f} Streetwear_FP={fp}"
        )

        if best is None or f1m > best["f1_macro"]:
            best = row | {"pred": pred.copy()}

    pd.DataFrame([{k: v for k, v in r.items() if k != "pred"} for r in search]).to_csv(
        RUN_DIR / "alpha_search.csv", index=False
    )

    p, r, f1, support = precision_recall_fscore_support(
        y_dev, best["pred"],
        labels=list(range(len(STYLE_LABELS))),
        zero_division=0,
    )
    pd.DataFrame({
        "style": STYLE_LABELS,
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": support,
    }).to_csv(RUN_DIR / "dev72_per_class.csv", index=False)

    np.savez_compressed(
        RUN_DIR / "final_prototypes.npz",
        image_prototypes=image_proto,
        text_prototypes=text_proto,
        alpha_text=np.array([best["alpha_text"]], dtype=np.float32),
        style_labels=np.array(STYLE_LABELS, dtype=object),
    )

    summary = {
        "model": MODEL_ID,
        "method": "FashionCLIP text + SFS real prototypes",
        "sfs_samples": len(sfs_rows),
        "dev_samples": len(dev_rows),
        "best_alpha_text": float(best["alpha_text"]),
        "best_alpha_sfs": float(best["alpha_sfs"]),
        "dev_accuracy": float(best["accuracy"]),
        "dev_f1_macro": float(best["f1_macro"]),
        "streetwear_false_positives": int(best["streetwear_false_positives"]),
    }
    (RUN_DIR / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== FASHIONCLIP + SFS PROTOTYPES ===")
    print(f"SFS samples={len(sfs_rows)}")
    print(f"Dev samples={len(dev_rows)}")
    print(f"Best mix: text={best['alpha_text']:.2f} SFS={best['alpha_sfs']:.2f}")
    print(f"Accuracy={best['accuracy']:.3f}")
    print(f"Macro-F1={best['f1_macro']:.3f}")
    print(f"Streetwear_FP={best['streetwear_false_positives']}")
    print("Per-class:", RUN_DIR / "dev72_per_class.csv")

if __name__ == "__main__":
    main()
