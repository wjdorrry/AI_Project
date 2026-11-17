from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BIG5_MANIFEST = ROOT / "dataset" / "week6_style_chunks" / "manifest.csv"
EXACT4_MANIFEST = ROOT / "dataset" / "week7_style_exact4" / "manifest.csv"

OUT_DIR = ROOT / "dataset" / "week7_style_balanced"
OUT_MANIFEST = OUT_DIR / "manifest.csv"

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

VAL_PER_CLASS = 13
TEST_PER_CLASS = 14
SEED = 20261117


def load_big5():
    rows = []
    with BIG5_MANIFEST.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            style = row["style"].strip()
            if style not in STYLE_LABELS:
                continue
            file_path = row["file"].replace("\\", "/")
            source_id = row["source_id"].strip()
            rows.append({
                "file": file_path,
                "style": style,
                "source": row.get("source", "shaghayegh/stylistic-bias-dataset"),
                "source_id": f"big5:{style}:{source_id}",
            })
    return rows


def load_exact4():
    rows = []
    with EXACT4_MANIFEST.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            style = row["style"].strip()
            if style not in STYLE_LABELS:
                continue
            rel = row["file"].replace("\\", "/")
            file_path = f"dataset/week7_style_exact4/{rel}"
            source_id = row["source_id"].strip()
            rows.append({
                "file": file_path,
                "style": style,
                "source": row.get("source", "lihicarmeli/fashion-stylist-multimodal-v2"),
                "source_id": f"exact4:{style}:{source_id}",
            })
    return rows


def main():
    if not BIG5_MANIFEST.exists():
        raise SystemExit(f"Не найден: {BIG5_MANIFEST}")
    if not EXACT4_MANIFEST.exists():
        raise SystemExit(f"Не найден: {EXACT4_MANIFEST}")

    rows = load_big5() + load_exact4()

    by_style = defaultdict(list)
    for row in rows:
        by_style[row["style"]].append(row)

    missing_styles = [s for s in STYLE_LABELS if not by_style[s]]
    if missing_styles:
        raise SystemExit(f"Нет данных для стилей: {missing_styles}")

    # Проверка дубликатов по source_id.
    all_ids = [r["source_id"] for r in rows]
    if len(all_ids) != len(set(all_ids)):
        raise SystemExit("Найдены повторяющиеся source_id. Останавливаю сборку.")

    final_rows = []
    rng = random.Random(SEED)

    for style in STYLE_LABELS:
        items = list(by_style[style])
        rng.shuffle(items)

        need = VAL_PER_CLASS + TEST_PER_CLASS
        if len(items) <= need:
            raise SystemExit(
                f"Недостаточно оригиналов для {style}: {len(items)}; нужно больше {need}."
            )

        val_items = items[:VAL_PER_CLASS]
        test_items = items[VAL_PER_CLASS:VAL_PER_CLASS + TEST_PER_CLASS]
        train_items = items[VAL_PER_CLASS + TEST_PER_CLASS:]

        for split, subset in (
            ("train", train_items),
            ("validation", val_items),
            ("test", test_items),
        ):
            for row in subset:
                out = dict(row)
                out["split"] = split
                final_rows.append(out)

    # Проверяем, что каждый файл реально существует.
    missing_files = []
    for row in final_rows:
        path = ROOT / row["file"]
        if not path.exists():
            missing_files.append(str(path))
            if len(missing_files) >= 10:
                break

    if missing_files:
        print("Не найдены некоторые изображения:")
        for p in missing_files:
            print("  ", p)
        raise SystemExit("Сборка остановлена: есть отсутствующие файлы.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUT_MANIFEST.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "style", "split", "source", "source_id"],
        )
        writer.writeheader()
        writer.writerows(final_rows)

    print("DONE")
    print("Manifest:", OUT_MANIFEST)
    print("Total originals:", len(final_rows))
    print()

    counts = Counter((r["style"], r["split"]) for r in final_rows)
    for style in STYLE_LABELS:
        train_n = counts[(style, "train")]
        val_n = counts[(style, "validation")]
        test_n = counts[(style, "test")]
        total_n = train_n + val_n + test_n
        print(
            f"{style:11s} total={total_n:3d} | "
            f"train={train_n:3d} val={val_n:2d} test={test_n:2d}"
        )

    print()
    print("Validation/test: только оригинальные изображения.")
    print("Следующий этап: balanced sampler для train без копирования файлов.")


if __name__ == "__main__":
    main()
