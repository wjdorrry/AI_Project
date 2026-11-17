from __future__ import annotations

import csv
import io
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARTS_DIR = ROOT / "dataset" / "sfs_parts"
METADATA = ROOT / "dataset" / "SFS_metadata.csv"

OLD_MANIFEST = ROOT / "dataset" / "sfs_needed_candidates" / "manifest.csv"
OUT_DIR = ROOT / "dataset" / "sfs_weak_candidates"
OUT_MANIFEST = OUT_DIR / "manifest.csv"

SOURCE_TO_TARGET = {
    "Elegant": "Elegant",
    "Basic": "Minimalist",
    "Sports": "Sporty",
    "Sporty": "Sporty",
    "Urban": "Streetwear",
    "Business": "Office",
    "Business Casual": "Office",
}

# Maximum NEW candidates to collect in this pass.
TARGET_LIMITS = {
    "Elegant": 300,
    "Minimalist": 300,
    "Sporty": 300,
    "Streetwear": 300,
    "Office": 300,
}


class SplitPartsReader(io.RawIOBase):
    def __init__(self, parts):
        super().__init__()
        self.parts = parts
        self.index = 0
        self.current = None
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


def load_metadata_targets():
    wanted = {}
    with METADATA.open("r", encoding="utf-8", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pic = str(row.get("pic_name", "")).strip()
            source_style = primary_style(str(row.get("styles", "")).strip())
            target = SOURCE_TO_TARGET.get(source_style)
            if pic and target:
                wanted[pic] = (target, source_style)
    return wanted


def load_seen_names():
    seen = set()

    for manifest in (OLD_MANIFEST, OUT_MANIFEST):
        if not manifest.exists():
            continue
        with manifest.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                name = row.get("pic_name") or Path(row.get("file", "")).name
                if name:
                    seen.add(name)

    return seen


def load_existing_rows():
    if not OUT_MANIFEST.exists():
        return []
    with OUT_MANIFEST.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    if not METADATA.exists():
        raise SystemExit(f"Не найден {METADATA}")

    parts = sorted(
        p for p in PARTS_DIR.glob("images-*")
        if p.is_file() and p.name.split("-")[-1].isdigit()
    )
    if len(parts) < 3:
        raise SystemExit("Ожидаются images-00, images-01 и images-02.")

    print("Части:")
    for p in parts:
        print(f"  {p.name}: {p.stat().st_size / (1024**3):.2f} GB")

    wanted = load_metadata_targets()
    seen = load_seen_names()
    rows = load_existing_rows()

    counts = Counter(r["style"] for r in rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for target in TARGET_LIMITS:
        (OUT_DIR / target).mkdir(parents=True, exist_ok=True)

    print("Уже известных pic_name:", len(seen))
    print("Уже сохранено в weak-наборе:", dict(counts))
    print("\nЧитаю split-архив...")

    reader = SplitPartsReader(parts)
    buffered = io.BufferedReader(reader, buffer_size=1024 * 1024)

    saved_now = 0
    seen_members = 0

    try:
        with tarfile.open(fileobj=buffered, mode="r|gz") as tf:
            for member in tf:
                seen_members += 1
                if not member.isfile():
                    continue

                base = Path(member.name).name
                info = wanted.get(base)
                if info is None or base in seen:
                    continue

                target, source_style = info
                if counts[target] >= TARGET_LIMITS[target]:
                    continue

                extracted = tf.extractfile(member)
                if extracted is None:
                    continue

                out_path = OUT_DIR / target / base
                with out_path.open("wb") as out:
                    while True:
                        chunk = extracted.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)

                rows.append({
                    "file": str(out_path.relative_to(ROOT)).replace("\\", "/"),
                    "style": target,
                    "source_style": source_style,
                    "pic_name": base,
                    "source": "SFS-weak-pass",
                })
                seen.add(base)
                counts[target] += 1
                saved_now += 1

                if saved_now % 25 == 0:
                    print(
                        f"Новых: {saved_now} | "
                        + ", ".join(f"{s}={counts[s]}" for s in TARGET_LIMITS)
                    )

                if all(counts[s] >= TARGET_LIMITS[s] for s in TARGET_LIMITS):
                    print("Достигнуты лимиты для всех слабых классов.")
                    break

    except (tarfile.ReadError, EOFError, OSError) as exc:
        print("\nДошли до конца скачанных частей.")
        print("Это ожидаемо для неполного split-архива.")
        print("Техническое сообщение:", type(exc).__name__, str(exc)[:160])
    finally:
        try:
            buffered.close()
        except Exception:
            pass

    with OUT_MANIFEST.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "style", "source_style", "pic_name", "source"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== SFS WEAK CLASSES EXTRACTION ===")
    print("Просмотрено элементов архива:", seen_members)
    print("Сохранено новых кандидатов:", saved_now)
    print("По слабым классам:")
    for target in TARGET_LIMITS:
        print(f"  {target:11s}: {counts[target]}")
    print("Manifest:", OUT_MANIFEST)
    print(
        "\nВАЖНО: это пока кандидаты. Следующим шагом прогоняем их "
        "через FashionCLIP; Basic→Minimalist снова не принимаем автоматически."
    )


if __name__ == "__main__":
    main()
