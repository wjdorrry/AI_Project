from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = ROOT / "dataset" / "week7_real_benchmark"
RUN_DIR = ROOT / "ml" / "runs" / "week7_fashionclip_zero_shot"
MODEL_ID = "patrickjohncyh/fashion-clip"

STYLE_LABELS = [
    "Elegant", "Casual", "Romantic", "Minimalist", "Sporty",
    "Streetwear", "Boho", "Office", "Evening",
]
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


def collect_rows():
    rows = []
    for style in STYLE_LABELS:
        folder = BENCHMARK_DIR / style
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


def save_confusion(y_true, y_pred, path):
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
        title="FashionCLIP zero-shot — real benchmark",
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


def main():
    rows = collect_rows()
    if not rows:
        raise SystemExit("Не найдены изображения в dataset/week7_real_benchmark/<STYLE>/")

    counts = pd.Series([r["style"] for r in rows]).value_counts()
    print("Найдено изображений:", len(rows))
    print("По стилям:", counts.to_dict())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Устройство:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Загрузка FashionCLIP:", MODEL_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotImageClassification.from_pretrained(MODEL_ID)
    model = model.to(device)
    model.eval()

    all_prompts = []
    prompt_style_ids = []
    for style_id, style in enumerate(STYLE_LABELS):
        for prompt in PROMPTS[style]:
            all_prompts.append(prompt)
            prompt_style_ids.append(style_id)

    y_true = []
    y_pred = []
    results = []

    with torch.no_grad():
        for index, row in enumerate(rows, start=1):
            image = Image.open(row["path"]).convert("RGB")
            inputs = processor(
                text=all_prompts,
                images=image,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            prompt_logits = outputs.logits_per_image[0]

            style_logits = []
            for style_id in range(len(STYLE_LABELS)):
                ids = [i for i, sid in enumerate(prompt_style_ids) if sid == style_id]
                style_logits.append(prompt_logits[ids].mean())
            style_logits = torch.stack(style_logits)
            probs = torch.softmax(style_logits, dim=0)
            top2 = torch.topk(probs, k=2)

            true_id = STYLE_LABELS.index(row["style"])
            pred_id = int(top2.indices[0].item())
            second_id = int(top2.indices[1].item())
            conf = float(top2.values[0].item())
            second_conf = float(top2.values[1].item())
            margin = conf - second_conf

            y_true.append(true_id)
            y_pred.append(pred_id)
            results.append({
                "file": row["file"],
                "true_style": row["style"],
                "pred_style": STYLE_LABELS[pred_id],
                "correct": pred_id == true_id,
                "confidence": conf,
                "second_style": STYLE_LABELS[second_id],
                "second_confidence": second_conf,
                "margin": margin,
            })

            print(
                f"[{index:02d}/{len(rows)}] {row['style']:11s} -> "
                f"{STYLE_LABELS[pred_id]:11s} conf={conf:.3f} margin={margin:.3f}"
            )

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    result_df = pd.DataFrame(results)
    result_df.to_csv(RUN_DIR / "predictions.csv", index=False)

    accuracy = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred,
        labels=list(range(len(STYLE_LABELS))),
        zero_division=0,
    )
    pd.DataFrame({
        "style": STYLE_LABELS,
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": support,
    }).to_csv(RUN_DIR / "per_class.csv", index=False)

    save_confusion(y_true, y_pred, RUN_DIR / "confusion_matrix.png")

    sid = STYLE_LABELS.index("Streetwear")
    streetwear_fp = sum(1 for y, p_ in zip(y_true, y_pred) if p_ == sid and y != sid)

    summary = {
        "model": MODEL_ID,
        "samples": len(rows),
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "streetwear_false_positives": streetwear_fp,
        "mean_confidence": float(result_df["confidence"].mean()),
        "mean_margin": float(result_df["margin"].mean()),
        "counts": {s: int(counts.get(s, 0)) for s in STYLE_LABELS},
    }
    (RUN_DIR / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== FASHIONCLIP ZERO-SHOT REAL BENCHMARK ===")
    print(f"Samples={len(rows)}")
    print(f"Accuracy={accuracy:.3f}")
    print(f"Macro-F1={f1_macro:.3f}")
    print(f"Streetwear_FP={streetwear_fp}")
    print(f"Mean confidence={summary['mean_confidence']:.3f}")
    print(f"Mean margin={summary['mean_margin']:.3f}")
    print("Результаты:", RUN_DIR)


if __name__ == "__main__":
    main()
