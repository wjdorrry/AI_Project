from __future__ import annotations

import csv
import io
import os
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARTS_DIR = ROOT / "dataset" / "sfs_parts"
METADATA = ROOT / "dataset" / "SFS_metadata.csv"
OUT_DIR = ROOT / "dataset" / "sfs_needed_candidates"
MANIFEST = OUT_DIR / "manifest.csv"

# SFS primary style -> our candidate class.
# Ambiguous mappings (Basic/Urban/etc.) are only candidates;
# they will be filtered with FashionCLIP later.
STYLE_MAP = {
    "Elegant": "Elegant",
    "Casual": "Casual",
    "Romantic": "Romantic",
    "Bohemian": "Boho",
    "Urban": "Streetwear",
    "Business": "Office",
    "Business Casual": "Office",
    "Formal": "Evening",
    "Sports": "Sporty",
    "Sporty": "Sporty",
    "Basic": "Minimalist",
}

# Keep disk use bounded. We can raise this later if needed.
PER_CLASS_LIMIT = int(os.getenv("SFS_PER_CLASS_LIMIT", "300"))

TARGETS = [
    "Elegant", "Casual", "Romantic", "Minimalist", "Sporty",
    "Streetwear", "Boho", "Office", "Evening",
]


class SplitPartsReader(io.RawIOBase):
    """Read images-00, images-01, ... sequentially as one binary stream."""

    def __init__(self, parts: list[Path]):
        super().__init__()
        self.parts = parts
        self.index = 0
        self.current = None
        self.total_read = 0
        self._open_current()

    def _open_current(self):
        if self.current is not None:
            self.current.close()
            self.current = None
        if self.index < len(self.parts):
            self.current = self.parts[self.index].open("rb")

    def readable(self):
        return True

    def readinto(self, b):
        if self.current is None:
            return 0

        mv = memoryview(b)
        written = 0

        while written < len(mv) and self.current is not None:
            n = self.current.readinto(mv[written:])
            if n:
                written += n
                self.total_read += n
                break

            self.index += 1
            self._open_current()

        return written

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def primary_style(value: str) -> str:
    if not value:
        return ""
    return value.split(",", 1)[0].strip()


def load_targets() -> dict[str, tuple[str, str]]:
    if not METADATA.exists():
        raise SystemExit(
            f"Не найден metadata CSV: {METADATA}\n"
            "Положите SFS_metadata.csv в папку dataset."
        )

    targets: dict[str, tuple[str, str]] = {}
    source_counts = Counter()

    with METADATA.open("r", encoding="utf-8", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        required = {"pic_name", "styles"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"В CSV нет колонок: {sorted(missing)}")

        for row in reader:
            pic = str(row.get("pic_name", "")).strip()
            src_style = primary_style(str(row.get("styles", "")).strip())
            mapped = STYLE_MAP.get(src_style)
            if not pic or not mapped:
                continue
            targets[pic] = (mapped, src_style)
            source_counts[src_style] += 1

    print("Кандидатов в metadata:", len(targets))
    print("По исходным стилям:", dict(source_counts))
    return targets


def main():
    parts = sorted(
        p for p in PARTS_DIR.glob("images-*")
        if p.is_file() and p.name.split("-")[-1].isdigit()
    )
    if not parts:
        raise SystemExit(f"Не найдены части images-* в {PARTS_DIR}")

    print("Части:")
    for p in parts:
        print(f"  {p.name}: {p.stat().st_size / (1024**3):.2f} GB")

    targets = load_targets()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for style in TARGETS:
        (OUT_DIR / style).mkdir(parents=True, exist_ok=True)

    existing = Counter()
    rows: list[dict[str, str]] = []

    if MANIFEST.exists():
        with MANIFEST.open("r", encoding="utf-8", newline="") as f:
            old = list(csv.DictReader(f))
        rows.extend(old)
        for r in old:
            existing[r["style"]] += 1
        print("Уже сохранено:", dict(existing))

    saved_names = {Path(r["file"]).name for r in rows}
    counts = Counter(existing)

    split_reader = SplitPartsReader(parts)
    buffered = io.BufferedReader(split_reader, buffer_size=1024 * 1024)

    seen_members = 0
    saved_now = 0

    print("\nЧитаю доступное начало split-архива...")
    try:
        with tarfile.open(fileobj=buffered, mode="r|gz") as tf:
            for member in tf:
                seen_members += 1
                if not member.isfile():
                    continue

                base = Path(member.name).name
                info = targets.get(base)
                if info is None:
                    continue

                mapped_style, src_style = info
                if counts[mapped_style] >= PER_CLASS_LIMIT:
                    continue

                safe_name = base
                if safe_name in saved_names:
                    continue

                extracted = tf.extractfile(member)
                if extracted is None:
                    continue

                out_path = OUT_DIR / mapped_style / safe_name
                with out_path.open("wb") as out:
                    while True:
                        chunk = extracted.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)

                rows.append({
                    "file": str(out_path.relative_to(ROOT)).replace("\\", "/"),
                    "style": mapped_style,
                    "source_style": src_style,
                    "pic_name": base,
                    "source": "SFS",
                })
                saved_names.add(base)
                counts[mapped_style] += 1
                saved_now += 1

                if saved_now % 25 == 0:
                    print(
                        f"Сохранено сейчас: {saved_now} | "
                        + ", ".join(f"{s}={counts[s]}" for s in TARGETS)
                    )

                if all(counts[s] >= PER_CLASS_LIMIT for s in TARGETS):
                    print("Достигнут лимит для всех 9 классов.")
                    break

    except (tarfile.ReadError, EOFError, OSError) as exc:
        # Expected when we intentionally have only the first N parts.
        print("\nДошли до конца скачанных частей.")
        print("Это ожидаемо для неполного split-архива.")
        print("Техническое сообщение:", type(exc).__name__, str(exc)[:160])
    finally:
        try:
            buffered.close()
        except Exception:
            pass

    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "style", "source_style", "pic_name", "source"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== SFS PARTIAL EXTRACTION ===")
    print("Просмотрено элементов архива:", seen_members)
    print("Сохранено новых нужных фото:", saved_now)
    print("Итого по нашим кандидатам:")
    for style in TARGETS:
        print(f"  {style:11s}: {counts[style]}")
    print("Manifest:", MANIFEST)
    print(
        "\nВАЖНО: Basic→Minimalist, Urban→Streetwear и другие сопоставления "
        "пока считаются кандидатами. Следующим шагом отфильтруем их FashionCLIP."
    )


if __name__ == "__main__":
    main()
