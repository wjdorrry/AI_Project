from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent

DEV_CACHE = ROOT / "ml" / "runs" / "week7_sfs_prototypes" / "dev72_embeddings.npz"
SFS_CACHE = ROOT / "ml" / "runs" / "week7_sfs_prototypes" / "sfs_embeddings.npz"
OLD_PROTO = ROOT / "ml" / "runs" / "week7_fashionclip_dev72" / "final_prototypes.npz"

RUN_DIR = ROOT / "ml" / "runs" / "week7_dev72_plus_sfs"

STYLE_LABELS = [
    "Elegant", "Casual", "Romantic", "Minimalist", "Sporty",
    "Streetwear", "Boho", "Office", "Evening",
]
STREETWEAR_ID = STYLE_LABELS.index("Streetwear")

TEXT_WEIGHTS = [0.65, 0.70, 0.75, 0.80, 0.85]
SFS_WEIGHTS = [0.00, 0.025, 0.05, 0.075, 0.10, 0.15]


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def class_prototypes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    protos = []
    for cid in range(len(STYLE_LABELS)):
        part = x[y == cid]
        if len(part) == 0:
            raise RuntimeError(f"Нет примеров для {STYLE_LABELS[cid]}")
        protos.append(normalize(part.mean(axis=0, keepdims=True))[0])
    return np.stack(protos, axis=0)


def build_loo_dev_prototypes(x: np.ndarray, y: np.ndarray):
    sums = []
    counts = []
    for cid in range(len(STYLE_LABELS)):
        part = x[y == cid]
        sums.append(part.sum(axis=0))
        counts.append(len(part))
    sums = np.stack(sums, axis=0)
    counts = np.array(counts)

    all_loo = []
    for i in range(len(x)):
        sample_protos = []
        for cid in range(len(STYLE_LABELS)):
            s = sums[cid].copy()
            n = counts[cid]
            if y[i] == cid:
                s = s - x[i]
                n -= 1
            if n <= 0:
                raise RuntimeError(f"Недостаточно dev-примеров для {STYLE_LABELS[cid]}")
            proto = normalize((s / n)[None, :])[0]
            sample_protos.append(proto)
        all_loo.append(np.stack(sample_protos, axis=0))
    return np.stack(all_loo, axis=0)


def main():
    for p in [DEV_CACHE, SFS_CACHE, OLD_PROTO]:
        if not p.exists():
            raise SystemExit(f"Не найден: {p}")

    RUN_DIR.mkdir(parents=True, exist_ok=True)

    dev = np.load(DEV_CACHE, allow_pickle=True)
    sfs = np.load(SFS_CACHE, allow_pickle=True)
    old = np.load(OLD_PROTO, allow_pickle=True)

    x_dev = normalize(dev["x"].astype(np.float32))
    y_dev = dev["y"].astype(np.int64)
    x_sfs = normalize(sfs["x"].astype(np.float32))
    y_sfs = sfs["y"].astype(np.int64)
    text_proto = normalize(old["text_prototypes"].astype(np.float32))

    print("Dev samples:", len(x_dev))
    print("SFS samples:", len(x_sfs))

    loo_dev_proto = build_loo_dev_prototypes(x_dev, y_dev)
    sfs_proto = class_prototypes(x_sfs, y_sfs)

    results = []
    best = None

    for wt in TEXT_WEIGHTS:
        for ws in SFS_WEIGHTS:
            wd = 1.0 - wt - ws
            if wd < -1e-9:
                continue
            wd = max(0.0, wd)

            preds = []
            for i, feat in enumerate(x_dev):
                text_scores = feat @ text_proto.T
                dev_scores = feat @ loo_dev_proto[i].T
                sfs_scores = feat @ sfs_proto.T

                scores = wt * text_scores + wd * dev_scores + ws * sfs_scores
                preds.append(int(np.argmax(scores)))

            pred = np.array(preds, dtype=np.int64)
            acc = float(accuracy_score(y_dev, pred))
            f1m = float(f1_score(y_dev, pred, average="macro", zero_division=0))
            fp = int(np.sum((pred == STREETWEAR_ID) & (y_dev != STREETWEAR_ID)))

            row = {
                "text_weight": wt,
                "dev72_weight": wd,
                "sfs_weight": ws,
                "accuracy": acc,
                "macro_f1": f1m,
                "streetwear_fp": fp,
            }
            results.append(row)

            print(
                f"text={wt:.3f} dev={wd:.3f} sfs={ws:.3f} | "
                f"acc={acc:.3f} F1={f1m:.3f} Streetwear_FP={fp}"
            )

            key = (f1m, acc, -fp)
            if best is None or key > best["key"]:
                best = {"key": key, "row": row, "pred": pred.copy()}

    pd.DataFrame(results).sort_values(
        ["macro_f1", "accuracy", "streetwear_fp"],
        ascending=[False, False, True],
    ).to_csv(RUN_DIR / "grid_results.csv", index=False)

    row = best["row"]
    pred = best["pred"]

    p, r, f1, support = precision_recall_fscore_support(
        y_dev, pred,
        labels=list(range(len(STYLE_LABELS))),
        zero_division=0,
    )
    pd.DataFrame({
        "style": STYLE_LABELS,
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": support,
    }).to_csv(RUN_DIR / "best_per_class.csv", index=False)

    summary = {
        "dev_samples": int(len(x_dev)),
        "sfs_samples": int(len(x_sfs)),
        "best_text_weight": float(row["text_weight"]),
        "best_dev72_weight": float(row["dev72_weight"]),
        "best_sfs_weight": float(row["sfs_weight"]),
        "accuracy": float(row["accuracy"]),
        "macro_f1": float(row["macro_f1"]),
        "streetwear_fp": int(row["streetwear_fp"]),
        "baseline_reference_macro_f1": 0.694,
    }
    (RUN_DIR / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== DEV72 + SFS WEIGHTED PROTOTYPES ===")
    print(
        f"Best mix: text={row['text_weight']:.3f} "
        f"dev72={row['dev72_weight']:.3f} "
        f"SFS={row['sfs_weight']:.3f}"
    )
    print(f"Accuracy={row['accuracy']:.3f}")
    print(f"Macro-F1={row['macro_f1']:.3f}")
    print(f"Streetwear_FP={row['streetwear_fp']}")
    print("Baseline Macro-F1=0.694")
    print("Results:", RUN_DIR / "grid_results.csv")


if __name__ == "__main__":
    main()
