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
REAL_DIR = ROOT / "dataset" / "week7_real_benchmark"
RUN_DIR = ROOT / "ml" / "runs" / "week7_fashionclip_fewshot"
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
        "a minimalist clean simple fashion outfit",
        "a minimal outfit with clean lines and neutral styling",
        "simple understated minimalist clothing",
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
            enc = processor(
                text=PROMPTS[style],
                return_tensors="pt",
                padding=True,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            out = model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            pooled = out.pooler_output
            feats = model.text_projection(pooled)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            proto = feats.mean(dim=0)
            proto = proto / proto.norm()
            protos.append(proto.float().cpu().numpy())
    return normalize(np.stack(protos, axis=0))


def loo_image_prototypes(x, y, leave_idx):
    protos = []
    for class_id in range(len(STYLE_LABELS)):
        idx = np.where((y == class_id) & (np.arange(len(y)) != leave_idx))[0]
        if len(idx) == 0:
            raise RuntimeError(f"Нет few-shot примеров для {STYLE_LABELS[class_id]}")
        proto = x[idx].mean(axis=0, keepdims=True)
        proto = normalize(proto)[0]
        protos.append(proto)
    return np.stack(protos, axis=0)


def class_metrics(y_true, y_pred):
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
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


def main():
    rows = collect_rows()
    if len(rows) < 18:
        raise SystemExit("Слишком мало реальных фото для few-shot проверки.")

    counts = pd.Series([r["style"] for r in rows]).value_counts()
    print("Реальных фото:", len(rows))
    print("По стилям:", counts.to_dict())

    if any(counts.get(s, 0) < 3 for s in STYLE_LABELS):
        raise SystemExit("Нужно минимум по 3 фото каждого стиля.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Устройство:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    print("FashionCLIP:", MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID).to(device)
    model.eval()

    cache = RUN_DIR / "real_dev_embeddings.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        x = data["x"]
        y = data["y"]
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

    search = []
    best = None

    for alpha in ALPHAS:
        preds = []
        confs = []
        margins = []

        for i in range(len(y)):
            image_proto = loo_image_prototypes(x, y, i)

            text_scores = x[i] @ text_proto.T
            image_scores = x[i] @ image_proto.T

            # alpha=1.0 -> only text zero-shot
            # alpha=0.0 -> only real-image prototypes
            scores = alpha * text_scores + (1.0 - alpha) * image_scores

            order = np.argsort(-scores)
            pred = int(order[0])
            preds.append(pred)

            shifted = scores - scores.max()
            probs = np.exp(shifted / 0.05)
            probs = probs / probs.sum()
            confs.append(float(probs[order[0]]))
            margins.append(float(probs[order[0]] - probs[order[1]]))

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
            "mean_confidence": float(np.mean(confs)),
            "mean_margin": float(np.mean(margins)),
        }
        search.append(row)
        print(
            f"alpha_text={alpha:.2f} | acc={acc:.3f} "
            f"F1={f1m:.3f} Streetwear_FP={fp}"
        )

        if best is None or f1m > best["f1_macro"]:
            best = row | {"preds": list(preds)}

    pd.DataFrame([{k: v for k, v in r.items() if k != "preds"} for r in search]).to_csv(
        RUN_DIR / "alpha_search.csv", index=False
    )

    best_preds = best["preds"]
    class_metrics(y, best_preds).to_csv(RUN_DIR / "loo_per_class.csv", index=False)

    result_rows = []
    for i, pred in enumerate(best_preds):
        result_rows.append({
            "file": files[i],
            "true_style": STYLE_LABELS[int(y[i])],
            "pred_style": STYLE_LABELS[int(pred)],
            "correct": bool(pred == y[i]),
        })
    pd.DataFrame(result_rows).to_csv(RUN_DIR / "loo_predictions.csv", index=False)

    # Final prototypes to use later on a NEW holdout set.
    final_image_proto = []
    for class_id in range(len(STYLE_LABELS)):
        proto = x[y == class_id].mean(axis=0, keepdims=True)
        final_image_proto.append(normalize(proto)[0])
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
        "method": "FashionCLIP text + real-image prototypes",
        "validation": "leave-one-out on 27 real development images",
        "samples": int(len(y)),
        "best_alpha_text": float(best["alpha_text"]),
        "best_alpha_image": float(best["alpha_image"]),
        "accuracy": float(best["accuracy"]),
        "f1_macro": float(best["f1_macro"]),
        "streetwear_false_positives": int(best["streetwear_false_positives"]),
        "note": "Final prototypes must be evaluated on a NEW unseen real holdout set.",
    }
    (RUN_DIR / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== FASHIONCLIP FEW-SHOT LOO ===")
    print(f"Samples={len(y)}")
    print(
        f"Best mix: text={best['alpha_text']:.2f} "
        f"image={best['alpha_image']:.2f}"
    )
    print(f"Accuracy={best['accuracy']:.3f}")
    print(f"Macro-F1={best['f1_macro']:.3f}")
    print(f"Streetwear_FP={best['streetwear_false_positives']}")
    print("Следующий обязательный этап: новый real holdout, не эти 27 фото.")
    print("Результаты:", RUN_DIR)


if __name__ == "__main__":
    main()
