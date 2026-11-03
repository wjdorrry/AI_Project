from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
MATRIX_PATH = ROOT / "sillage-style" / "src" / "data" / "perfumeStyleMatrix.json"
OUT_DIR = ROOT / "analysis" / "week6"

FACTORS = ["Black", "Elegant", "Winter"]


def main() -> None:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    rows = {row["tag"]: row for row in matrix["tags"]}
    totals = {family: 0 for family in matrix["families"]}

    for tag in FACTORS:
        for family in matrix["families"]:
            totals[family] += int(rows[tag]["scores"].get(family, 0))

    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    winner, total = ranked[0]

    contributions = [
        {
            "factor": tag,
            "category": rows[tag].get("categoryRu", "Фактор"),
            "score": int(rows[tag]["scores"].get(winner, 0)),
        }
        for tag in FACTORS
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "factors": FACTORS,
        "winner": winner,
        "total_score": total,
        "contributions": contributions,
        "ranking": [{"family": family, "score": score} for family, score in ranked],
    }
    (OUT_DIR / "explanation_example.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [item["factor"] for item in contributions]
    values = [item["score"] for item in contributions]
    ax.bar(labels, values)
    ax.axhline(0, linewidth=1)
    ax.set_title(f"Вклад факторов в семейство {winner}")
    ax.set_xlabel("Фактор")
    ax.set_ylabel("Вес")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "winner_contributions.png", dpi=180)
    plt.close(fig)

    print("Готово.")
    print(f"Факторы: {', '.join(FACTORS)}")
    print(f"Победитель: {winner}, итоговый балл: {total}")
    for item in contributions:
        sign = "+" if item["score"] > 0 else ""
        print(f"  {item['category']} {item['factor']}: {sign}{item['score']}")
    print(f"JSON: {OUT_DIR / 'explanation_example.json'}")
    print(f"График: {OUT_DIR / 'winner_contributions.png'}")


if __name__ == "__main__":
    main()
