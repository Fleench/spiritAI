"""Download theological source texts used to build the SpiritAI corpus.

This downloader intentionally fetches both requested public-domain/reference
links instead of overwriting one URL with the other. Put the resulting files
through prepare_data.py after you combine/curate them into a single raw text
file for training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import requests

from paths import workspace_path

SOURCES = [
    {
        "name": "ante_nicene_fathers_complete",
        "url": "https://archive.org/stream/AnteNiceneFathersCompleteVolumesIToIX_201407/Ante-nicene%20fathers%20-%20complete%20volumes%20I%20to%20IX_djvu.txt",
    },
    {
        "name": "catechism_of_the_catholic_church_usccb",
        "url": "https://archive.org/stream/catechismofthecatholicchurch/Catechism%20of%20the%20Catholic%20Church%20-%20USCCB_djvu.txt",
    },
]

HEADER_RE = re.compile(r"^[A-Z][A-Z0-9 '\-:;,.]{5,}$")


def safe_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return text[:90] or "section"


def split_sections(text: str, output_dir: Path, source_name: str) -> int:
    section_dir = output_dir / f"{source_name}_sections"
    section_dir.mkdir(parents=True, exist_ok=True)
    current_title = source_name
    current_section: list[str] = []
    saved = 0

    def flush() -> None:
        nonlocal saved, current_section, current_title
        if len(current_section) < 50:
            return
        filename = section_dir / f"{saved:04d}_{safe_filename(current_title)}.txt"
        filename.write_text("\n".join(current_section), encoding="utf-8")
        saved += 1

    for line in text.splitlines():
        stripped = line.strip()
        looks_like_header = (
            stripped
            and len(stripped) > 5
            and len(stripped.split()) < 14
            and HEADER_RE.match(stripped) is not None
        )
        if looks_like_header:
            flush()
            current_title = stripped
            current_section = [line]
        elif current_section or stripped:
            current_section.append(line)
    flush()
    return saved


def download_source(url: str, timeout: int) -> str:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    response.encoding = response.encoding or "utf-8"
    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SpiritAI theological source texts.")
    parser.add_argument("--output-dir", default=workspace_path("raw_data"))
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--split-sections", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_parts: list[str] = []

    for source in SOURCES:
        print(f"Downloading {source['name']}...")
        text = download_source(source["url"], args.timeout)
        path = output_dir / f"{source['name']}.txt"
        path.write_text(text, encoding="utf-8")
        combined_parts.append(f"\n\n===== {source['name']} =====\n\n{text}")
        print(f"Saved {len(text):,} characters to {path}")
        if args.split_sections:
            count = split_sections(text, output_dir, source["name"])
            print(f"Saved {count:,} section files for {source['name']}")

    combined_path = output_dir / "theology_sources_combined.txt"
    combined_path.write_text("".join(combined_parts).strip(), encoding="utf-8")
    print(f"Combined corpus written to {combined_path}")
    print(f"Next: python prepare_data.py --input {workspace_path('raw_data', 'theology_sources_combined.txt')}")


if __name__ == "__main__":
    main()
