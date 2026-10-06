"""One-shot export: PerfumeStyleMatrix.xlsx -> perfumeStyleMatrix.json"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "PerfumeStyleMatrix.xlsx"
OUT = ROOT / "data" / "perfumeStyleMatrix.json"


def main() -> None:
    import openpyxl

    wb = openpyxl.load_workbook(XLSX, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        raise SystemExit("empty sheet")

    header = rows[0]
    families = [str(c).strip() for c in header[2:] if c is not None]

    tags = []
    for row in rows[1:]:
        if not row or row[0] is None:
            continue
        scores = {}
        for i, fam in enumerate(families):
            cell = row[i + 2] if i + 2 < len(row) else None
            scores[fam] = int(cell) if cell is not None else 0
        tags.append(
            {
                "tag": str(row[0]).strip(),
                "categoryRu": str(row[1]).strip() if row[1] is not None else "",
                "scores": scores,
            }
        )

    payload = {
        "meta": {
            "source": "PerfumeStyleMatrix.xlsx",
            "description": "Веса соответствия тегов одежды/стиля парфюмерным семействам (-10..10).",
        },
        "families": families,
        "tags": tags,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT} ({len(tags)} tags, {len(families)} families)")


if __name__ == "__main__":
    main()
