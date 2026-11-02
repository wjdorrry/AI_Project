from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
MATRIX_PATH = ROOT / "sillage-style" / "src" / "data" / "perfumeStyleMatrix.json"
OUT_DIR = ROOT / "analysis" / "week5"

LOOK_TAGS = ["Black", "Elegant"]
SEASONS = ["Spring", "Summer", "Autumn", "Winter"]


def score(matrix: dict, tags: list[str]) -> dict[str, int]:
    families = matrix["families"]
    rows = {row["tag"]: row["scores"] for row in matrix["tags"]}
    totals = {family: 0 for family in families}

    for tag in tags:
        row = rows[tag]
        for family in families:
            totals[family] += int(row.get(family, 0))

    return totals


def main() -> None:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_scores = {}
    for season in SEASONS:
        all_scores[season] = score(matrix, LOOK_TAGS + [season])

    csv_path = OUT_DIR / "season_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["family", *SEASONS])
        for family in matrix["families"]:
            writer.writerow([family, *[all_scores[s][family] for s in SEASONS]])

    fig, ax = plt.subplots(figsize=(12, 6))
    x = list(range(len(matrix["families"])))
    width = 0.2

    for index, season in enumerate(SEASONS):
        offset = (index - 1.5) * width
        ax.bar(
            [value + offset for value in x],
            [all_scores[season][family] for family in matrix["families"]],
            width=width,
            label=season,
        )

    ax.set_title("Влияние сезона на подбор семейств: Black + Elegant")
    ax.set_xlabel("Парфюмерное семейство")
    ax.set_ylabel("Итоговый балл")
    ax.set_xticks(x)
    ax.set_xticklabels(matrix["families"], rotation=35, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "season_comparison.png", dpi=180)
    plt.close(fig)

    print("Готово.")
    print(f"CSV: {csv_path}")
    print(f"График: {OUT_DIR / 'season_comparison.png'}")
    for season in SEASONS:
        ranked = sorted(all_scores[season].items(), key=lambda item: item[1], reverse=True)[:3]
        print(season, "->", ", ".join(f"{family}: {value}" for family, value in ranked))


if __name__ == "__main__":
    main()
