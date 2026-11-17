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
SOURCE_DIRS = [
    ROOT / "dataset" / "week7_real_benchmark",
    ROOT / "dataset" / "week7_real_holdout",
]
RUN_DIR = ROOT / "ml" / "runs" / "week7_fashionclip_dev72"
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


def collect_rows():
    rows = []
    for base in SOURCE_DIRS:
        if not base.exists():
            continue
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


def normalize(x):
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


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


def loo_image_prototypes(x, y, leave_idx):
    protos = []
    indexes = np.arange(len(y))
    for class_id in range(len(STYLE_LABELS)):
        idx = np.where((y == class_id) & (indexes != leave_idx))[0]
        proto = normalize(x[idx].mean(axis=0, keepdims=True))[0]
        protos.append(proto)
    return np.stack(protos, axis=0)


def main():
    rows = collect_rows()
    counts = pd.Series([r["style"] for r in rows]).value_counts()
    print("Real development фото:", len(rows))
    print("По стилям:", counts.to_dict())

    if len(rows) != 72 or any(counts.get(s, 0) != 8 for s in STYLE_LABELS):
        print("ВНИМАНИЕ: ожидается 72 фото, по 8 на каждый стиль.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Устройство:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    model.eval()

    cache = RUN_DIR / "real_dev72_embeddings.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        x, y = data["x"], data["y"]
        files = data["files"].tolist()
        print("Использую сохранённые embeddings:", len(y))
    else:
        feats, labels, files = [], [], []
        for i, row in enumerate(rows, start=1):
            image = Image.open(row["path"]).convert("RGB")
            feats.append(image_embedding(model, processor, image, device))
            labels.append(row["label"])
            files.append(row["file"])
            print(f"[{i:02d}/{len(rows)}] embedding {row['style']}")
        x = normalize(np.stack(feats, axis=0).astype(np.float32))
        y = np.array(labels, dtype=np.int64)
        np.savez_compressed(cache, x=x, y=y, files=np.array(files, dtype=object))

    text_proto = text_prototypes(model, processor, device)

    best = None
    search_rows = []

    for alpha in ALPHAS:
        preds = []
        for i in range(len(y)):
            image_proto = loo_image_prototypes(x, y, i)
            scores = alpha * (x[i] @ text_proto.T) + (1.0 - alpha) * (x[i] @ image_proto.T)
            preds.append(int(np.argmax(scores)))

        acc = float(accuracy_score(y, preds))
        f1m = float(f1_score(y, preds, average="macro", zero_division=0))
        sid = STYLE2ID["Streetwear"]
        fp = int(sum(1 for yt, yp in zip(y, preds) if yp == sid and yt != sid))

        row = {
            "alpha_text": alpha,
            "alpha_image": 1.0 - alpha,
            "accuracy": acc,
            "f1_macro": f1m,
            "streetwear_false_positives": fp,
        }
        search_rows.append(row)
        print(
            f"alpha_text={alpha:.2f} | acc={acc:.3f} "
            f"F1={f1m:.3f} Streetwear_FP={fp}"
        )

        if best is None or f1m > best["f1_macro"]:
            best = row | {"preds": preds}

    pd.DataFrame(search_rows).to_csv(RUN_DIR / "alpha_search.csv", index=False)

    p, r, f1, support = precision_recall_fscore_support(
        y, best["preds"], labels=list(range(len(STYLE_LABELS))), zero_division=0
    )
    pd.DataFrame({
        "style": STYLE_LABELS,
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": support,
    }).to_csv(RUN_DIR / "loo_per_class.csv", index=False)

    final_image_proto = []
    for class_id in range(len(STYLE_LABELS)):
        proto = normalize(x[y == class_id].mean(axis=0, keepdims=True))[0]
        final_image_proto.append(proto)
    final_image_proto = np.stack(final_image_proto, axis=0)

    np.savez_compressed(
        RUN_DIR / "final_prototypes.npz",
        image_prototypes=final_image_proto,
        text_prototypes=text_proto,
        alpha_text=np.array([best["alpha_text"]], dtype=np.float32),
        style_labels=np.array(STYLE_LABELS, dtype=object),
    )

    summary = {
        "model": MODEL_ID,
        "method": "FashionCLIP text + real image prototypes",
        "development_samples": int(len(y)),
        "samples_per_style": {s: int(counts.get(s, 0)) for s in STYLE_LABELS},
        "best_alpha_text": float(best["alpha_text"]),
        "best_alpha_image": float(best["alpha_image"]),
        "loo_accuracy": float(best["accuracy"]),
        "loo_f1_macro": float(best["f1_macro"]),
        "streetwear_false_positives": int(best["streetwear_false_positives"]),
        "note": "Must be evaluated on a new unseen final real set.",
    }
    (RUN_DIR / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== FASHIONCLIP REAL DEV72 LOO ===")
    print(f"Samples={len(y)}")
    print(f"Best mix: text={best['alpha_text']:.2f} image={best['alpha_image']:.2f}")
    print(f"Accuracy={best['accuracy']:.3f}")
    print(f"Macro-F1={best['f1_macro']:.3f}")
    print(f"Streetwear_FP={best['streetwear_false_positives']}")
    print("Prototypes:", RUN_DIR / "final_prototypes.npz")


if __name__ == "__main__":
    main()
