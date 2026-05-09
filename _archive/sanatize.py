"""Backward-compatible wrapper around prepare_data.py sanitation helpers.

The project now uses prepare_data.py to sanitize, deduplicate, tokenize, and
write train.bin/val.bin. This wrapper keeps the old script name working for
cleaning files into the workspace data directory without tokenization.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

from paths import workspace_path
from prepare_data import deduplicate_paragraphs, sanitize_text

RAW_DATA_DIR = workspace_path("raw_data")
CLEAN_DATA_DIR = workspace_path("data")


def clean_text(text: str) -> str:
    clean, _ = deduplicate_paragraphs(sanitize_text(text))
    return clean


def process_files() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Scanning for raw files in {RAW_DATA_DIR}...")

    raw_files = glob.glob(str(RAW_DATA_DIR / "*.txt")) + glob.glob(str(RAW_DATA_DIR / "*.md"))
    if not raw_files:
        print(f"No .txt or .md files found in {RAW_DATA_DIR}.")

    for filepath in raw_files:
        source_path = Path(filepath)
        print(f"Sanitizing: {source_path.name}...")
        clean_content = clean_text(source_path.read_text(encoding="utf-8"))
        (CLEAN_DATA_DIR / source_path.name).write_text(clean_content, encoding="utf-8")

    json_path = RAW_DATA_DIR / "CPDV.json"
    if json_path.exists():
        print("Sanitizing: CPDV.json...")
        data = json.loads(json_path.read_text(encoding="utf-8"))
        for book in data.get("books", []):
            for chapter in book.get("chapters", []):
                for verse in chapter.get("verses", []):
                    verse["text"] = clean_text(verse["text"])
        (CLEAN_DATA_DIR / "CPDV.json").write_text(json.dumps(data, indent=4), encoding="utf-8")

    print("Sanitization complete. Next: python prepare_data.py")
    print("prepare_data.py will auto-detect the cleaned files in the data directory.")


if __name__ == "__main__":
    process_files()
