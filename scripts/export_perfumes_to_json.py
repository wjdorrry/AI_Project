"""
Экспорт спарсенной базы духов (xlsx) → JSON для фронта Sillage&Style.

Положите файл perfume_database_cleaned.xlsx в корень проекта (рядом с PerfumeStyleMatrix.xlsx)
или укажите путь:  py export_perfumes_to_json.py --xlsx "C:\\path\\file.xlsx"

Колонки определяются по заголовкам (бренд/название/семейство/ноты). Семейства приводятся
к ключам матрицы: Citrus, Aquatic, Green, Fruity, Woody, Musk, Powdery, Gourmand,
Oriental, Floral, Leather.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XLSX = ROOT / "perfume_database_cleaned.xlsx"
# В public/ — Vite отдаёт как статику; не раздувает JS-бандл.
DEFAULT_OUT = ROOT / "sillage-style" / "public" / "perfumes.json"

# Подстроки в заголовках (нижний регистр) → логическое поле
HEADER_HINTS: dict[str, tuple[str, ...]] = {
    "brand": ("brand", "бренд", "house", "марка", "дизайнер", "designer"),
    "name": ("name", "название", "perfume", "title", "аромат", "fragrance"),
    "family_text": (
        "family",
        "семейств",
        "олфакт",
        "группа",
        "accord",
        "аккорд",
        "класс",
        "type",
        "категория",
        "pyramid",
    ),
    "notes": ("notes", "ноты", "описание", "description", "composition", "состав"),
}

MATRIX_FAMILIES: tuple[str, ...] = (
    "Citrus",
    "Aquatic",
    "Green",
    "Fruity",
    "Woody",
    "Musk",
    "Powdery",
    "Gourmand",
    "Oriental",
    "Floral",
    "Leather",
)

# (подстрока в нижнем регистре, без учёта регистра) → ключ семейства
FAMILY_SNIPPETS: tuple[tuple[str, str], ...] = (
    ("citrus", "Citrus"),
    ("цитрус", "Citrus"),
    ("цитрусов", "Citrus"),
    ("aquatic", "Aquatic"),
    ("акват", "Aquatic"),
    ("водн", "Aquatic"),
    ("морск", "Aquatic"),
    ("marine", "Aquatic"),
    ("green", "Green"),
    ("зелен", "Green"),
    ("трав", "Green"),
    ("herb", "Green"),
    ("fruity", "Fruity"),
    ("фрукт", "Fruity"),
    ("ягод", "Fruity"),
    ("wood", "Woody"),
    ("древес", "Woody"),
    ("woody", "Woody"),
    ("кедр", "Woody"),
    ("сандал", "Woody"),
    ("musk", "Musk"),
    ("муск", "Musk"),
    ("powder", "Powdery"),
    ("пудр", "Powdery"),
    ("ирис", "Powdery"),
    ("gourmand", "Gourmand"),
    ("гурман", "Gourmand"),
    ("ваниль", "Gourmand"),
    ("oriental", "Oriental"),
    ("восточ", "Oriental"),
    ("амбр", "Oriental"),
    ("пряност", "Oriental"),
    ("floral", "Floral"),
    ("цветоч", "Floral"),
    ("флорал", "Floral"),
    ("rose", "Floral"),
    ("роза", "Floral"),
    ("jasmin", "Floral"),
    ("жасмин", "Floral"),
    ("leather", "Leather"),
    ("кож", "Leather"),
    ("шкур", "Leather"),
)


def _norm_header(s: object) -> str:
    if s is None:
        return ""
    t = str(s).strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def _guess_columns(headers: list[str]) -> dict[str, int | None]:
    nh = [_norm_header(h) for h in headers]
    out: dict[str, int | None] = {k: None for k in HEADER_HINTS}
    for i, h in enumerate(nh):
        for field, hints in HEADER_HINTS.items():
            if out[field] is not None:
                continue
            for hint in hints:
                if hint in h:
                    out[field] = i
                    break
    return out


def _families_from_blob(blob: str) -> list[str]:
    if not blob or not blob.strip():
        return []
    low = blob.lower()
    found: set[str] = set()
    for needle, fam in FAMILY_SNIPPETS:
        if needle in low:
            found.add(fam)
    # точное совпадение с англ. ключом
    for fam in MATRIX_FAMILIES:
        if fam.lower() in low:
            found.add(fam)
    return sorted(found, key=lambda f: MATRIX_FAMILIES.index(f))


def _cell(row: tuple[object, ...], idx: int | None) -> str:
    if idx is None or idx >= len(row):
        return ""
    v = row[idx]
    if v is None:
        return ""
    return str(v).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export perfume xlsx to JSON for sillage-style")
    parser.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX, help="Path to cleaned perfume database xlsx")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output perfumes.json")
    args = parser.parse_args()

    if not args.xlsx.exists():
        raise SystemExit(
            f"Файл не найден: {args.xlsx}\n"
            "Скопируйте perfume_database_cleaned.xlsx в корень проекта или укажите --xlsx путь."
        )

    import openpyxl

    wb = openpyxl.load_workbook(args.xlsx, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        raise SystemExit("empty sheet")

    header_row = rows[0]
    headers = [str(c).strip() if c is not None else "" for c in header_row]
    colmap = _guess_columns(headers)

    perfumes: list[dict[str, object]] = []
    for r_i, row in enumerate(rows[1:], start=1):
        if not row or all(c is None or str(c).strip() == "" for c in row):
            continue
        brand = _cell(row, colmap["brand"])
        name = _cell(row, colmap["name"])
        fam_col = _cell(row, colmap["family_text"])
        notes_raw = _cell(row, colmap["notes"])
        # если нет отдельной колонки семейства — ищем ключи во всей строке
        blob = " | ".join(str(c) for c in row if c is not None and str(c).strip())
        families = _families_from_blob(fam_col if fam_col else blob)
        if not brand and not name:
            continue
        if not families:
            continue
        notes_trim = 140
        notes_short = (notes_raw[:notes_trim] + ("…" if len(notes_raw) > notes_trim else "")) if notes_raw else ""
        perfumes.append(
            {
                "id": str(r_i),
                "brand": brand or "—",
                "name": name or "—",
                "notes": notes_short,
                "families": families,
            }
        )

    payload = {
        "meta": {
            "source": str(args.xlsx.name),
            "description": "Справочник для подсказок; families — ключи как в perfumeStyleMatrix.json",
            "columns_detected": {k: headers[i] if i is not None and i < len(headers) else None for k, i in colmap.items()},
        },
        "perfumes": perfumes,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Без отступов — файл меньше, парсится быстрее в браузере.
    args.out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {args.out} ({len(perfumes)} perfumes with families)")


if __name__ == "__main__":
    main()
