"""Sanitize theological text and encode it into Nano-GPT train.bin/val.bin files.

Usage:
    python prepare_data.py
    python prepare_data.py --input $WORKSPACE_PATH/raw_data/theology.txt --output-dir $WORKSPACE_PATH/data

If --input is omitted, the script auto-detects cleaned .txt/.md files in
the output directory (for example files produced by sanatize.py) and combines
them for tokenization.

The output directory receives:
    clean.txt      deduplicated, normalized text
    train.bin      int32 tiktoken/gpt2 token ids for training
    val.bin        int32 tiktoken/gpt2 token ids for validation
    meta.json      run metadata

No custom vocabulary is generated. Training must use the standard tiktoken gpt2
vocabulary size of 50,257.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import tiktoken

from paths import workspace_path

DEFAULT_RAW_DATA_DIR = workspace_path("raw_data")
GPT2_VOCAB_SIZE = 50_257
SPLIT_WORD_FIXES = {
    r"\bj\s+oined\b": "joined",
    r"\bjo\s+ined\b": "joined",
    r"\bth\s+e\b": "the",
    r"\bt\s+he\b": "the",
    r"\bw\s+ith\b": "with",
    r"\ba\s+nd\b": "and",
    r"\bo\s+f\b": "of",
    r"\bi\s+n\b": "in",
    r"\bt\s+o\b": "to",
    r"\bf\s+or\b": "for",
    r"\bL\s+ord\b": "Lord",
    r"\bG\s+od\b": "God",
    r"\bC\s+hrist\b": "Christ",
    r"\bS\s+pirit\b": "Spirit",
    r"\bC\s+hurch\b": "Church",
}


def fix_ocr_split_words(text: str) -> str:
    """Repair common OCR split-word artifacts without merging normal prose."""
    for pattern, replacement in SPLIT_WORD_FIXES.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Join words made entirely of separated letters, e.g. "c h u r c h".
    text = re.sub(
        r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b",
        lambda match: match.group(0).replace(" ", ""),
        text,
    )
    return text


def sanitize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"([A-Za-z])[-¬]\n\s*([A-Za-z])", r"\1\2", text)
    text = fix_ocr_split_words(text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"(\d+)\s*:\s*(\d+)", r"\1:\2", text)
    text = re.sub(r"[\^\*\~_=#\+]{2,}", " ", text)
    text = re.sub(r"[^\w\s\.,;:'\"!?\-()\[\]{}]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def paragraph_fingerprint(paragraph: str) -> str:
    normalized = re.sub(r"\W+", " ", paragraph.casefold()).strip()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def deduplicate_paragraphs(text: str, min_chars: int = 40) -> tuple[str, int]:
    paragraphs = re.split(r"\n\s*\n", text)
    seen: set[str] = set()
    kept: list[str] = []
    removed = 0
    for paragraph in paragraphs:
        paragraph = re.sub(r"[ \t]*\n[ \t]*", " ", paragraph).strip()
        if not paragraph:
            continue
        if len(paragraph) >= min_chars:
            fp = paragraph_fingerprint(paragraph)
            if fp in seen:
                removed += 1
                continue
            seen.add(fp)
        kept.append(paragraph)
    return "\n\n".join(kept), removed


def resolve_input_paths(input_args: list[str] | None, output_dir: Path) -> list[Path]:
    """Resolve explicit input files or auto-detect cleaned corpus files."""
    if input_args:
        return [Path(path) for path in input_args]

    candidate_dirs = []
    for directory in (output_dir, Path("data"), DEFAULT_RAW_DATA_DIR):
        if directory not in candidate_dirs:
            candidate_dirs.append(directory)

    preferred_names = (
        "clean.txt",
        "chat_corpus.txt",
        "theology_sources_combined.txt",
        "theology.txt",
    )
    for directory in candidate_dirs:
        for name in preferred_names:
            path = directory / name
            if path.exists():
                return [path]

    for directory in candidate_dirs:
        if not directory.exists():
            continue
        paths = sorted(
            path
            for pattern in ("*.txt", "*.md")
            for path in directory.glob(pattern)
            if path.name not in {"auto_corpus.txt"}
        )
        if paths:
            return paths

    searched = ", ".join(str(directory) for directory in candidate_dirs)
    raise FileNotFoundError(
        "No input corpus found. Run data.py and/or sanatize.py first, or pass "
        f"--input /path/to/corpus.txt. Searched: {searched}"
    )


def read_corpus(input_paths: list[Path]) -> str:
    """Read one or more UTF-8 text/markdown files into a single corpus."""
    missing = [str(path) for path in input_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Input file(s) not found: {', '.join(missing)}")

    corpus_parts: list[str] = []
    for path in input_paths:
        corpus_parts.append(f"\n\n===== {path} =====\n\n{path.read_text(encoding='utf-8')}")
    return "".join(corpus_parts).strip()


def encode_gpt2_int32(text: str) -> np.ndarray:
    """Encode text with tiktoken's GPT-2 BPE and return disk-safe int32 ids."""
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode(text)
    return np.asarray(ids, dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare sanitized tiktoken/gpt2 token bins for Nano-GPT training.")
    parser.add_argument(
        "--input",
        nargs="*",
        help=(
            "Optional raw UTF-8 text/markdown file(s) to sanitize and tokenize. "
            "If omitted, cleaned corpus files are auto-detected."
        ),
    )
    parser.add_argument("--output-dir", default=workspace_path("data"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be greater than 0 and less than 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = resolve_input_paths(args.input, output_dir)

    print("Preparing corpus from:")
    for path in input_paths:
        print(f"  - {path}")

    raw_text = read_corpus(input_paths)
    clean_text = sanitize_text(raw_text)
    clean_text, duplicates_removed = deduplicate_paragraphs(clean_text)
    (output_dir / "clean.txt").write_text(clean_text, encoding="utf-8")

    ids = encode_gpt2_int32(clean_text)
    if len(ids) < 10_000:
        raise ValueError(f"Only {len(ids)} tokens found; expected a much larger training corpus.")

    split_idx = int((1.0 - args.val_fraction) * len(ids))
    train_ids = ids[:split_idx].astype(np.int32, copy=False)
    val_ids = ids[split_idx:].astype(np.int32, copy=False)
    train_ids.tofile(output_dir / "train.bin")
    val_ids.tofile(output_dir / "val.bin")

    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": [str(path) for path in input_paths],
                "raw_chars": len(raw_text),
                "clean_chars": len(clean_text),
                "duplicates_removed": duplicates_removed,
                "tokens": int(len(ids)),
                "train_tokens": int(len(train_ids)),
                "val_tokens": int(len(val_ids)),
                "encoding": "gpt2",
                "vocab_size": GPT2_VOCAB_SIZE,
                "dtype": "int32",
            },
            f,
            indent=2,
        )

    print(f"Wrote {len(train_ids):,} train tokens and {len(val_ids):,} val tokens to {output_dir}")
    print(f"Encoding: tiktoken/gpt2; vocab size: {GPT2_VOCAB_SIZE:,}; dtype: int32")
    print(f"Duplicate paragraphs removed: {duplicates_removed:,}")


if __name__ == "__main__":
    main()
