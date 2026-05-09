"""Data processing pipeline for SpiritAI.

This module sanitizes, deduplicates, weights, and tokenizes raw datasets,
saving them into train.bin and val.bin formats compatible with NanoGPT.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Iterator

import numpy as np
import tiktoken

from spirit.config import DATA_DIR, RAW_DATA_DIR, TRAIN_BIN_PATH, VAL_BIN_PATH

logger = logging.getLogger(__name__)


def sanitize_text(text: str) -> str:
    """Sanitize input text by normalizing unicode and stripping excessive whitespace.

    Args:
        text: The raw text string.

    Returns:
        The sanitized text string.
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def deduplicate(lines: list[str]) -> list[str]:
    """Remove duplicate text segments.

    Args:
        lines: A list of text segments.

    Returns:
        A list of unique text segments, preserving order.
    """
    seen = set()
    unique = []
    for line in lines:
        if not line:
            continue
        hashed = hash(line)
        if hashed not in seen:
            seen.add(hashed)
            unique.append(line)
    return unique


def load_raw_file(filename: str) -> list[str]:
    """Load and split a raw text file by double newlines.

    Args:
        filename: Name of the file in the raw data directory.

    Returns:
        A list of text segments.
    """
    path = RAW_DATA_DIR / filename
    if not path.exists():
        logger.warning(f"File {path} not found. Skipping.")
        return []

    text = path.read_text(encoding="utf-8")
    return text.split("\n\n")


def prepare_dataset() -> None:
    """Sanitize, deduplicate, weight, and tokenize the datasets."""
    logger.info("Preparing dataset pipeline...")

    # Load sources
    anf_segments = load_raw_file("ante_nicene_fathers.txt")
    bible_segments = load_raw_file("bible.txt")
    alpaca_turns = load_raw_file("alpaca.txt")
    theo_turns = load_raw_file("theological_qa.txt")
    wiki_articles = load_raw_file("wikipedia.txt")

    # Sanitize and deduplicate
    anf_segments = deduplicate([sanitize_text(t) for t in anf_segments])
    bible_segments = deduplicate([sanitize_text(t) for t in bible_segments])
    alpaca_turns = deduplicate([sanitize_text(t) for t in alpaca_turns])
    theo_turns = deduplicate([sanitize_text(t) for t in theo_turns])
    wiki_articles = deduplicate([sanitize_text(t) for t in wiki_articles])

    # Weight mixing
    # Theological Q&A repeated 3x
    # HuggingFace (Alpaca) 1x
    # Wikipedia filtered and capped (already handled in source, 1x here)
    # Background theological texts (ANF, Bible) 1x
    combined_segments = []

    combined_segments.extend(anf_segments)
    combined_segments.extend(bible_segments)
    combined_segments.extend(alpaca_turns)
    combined_segments.extend(wiki_articles)

    for _ in range(3):
        combined_segments.extend(theo_turns)

    # Join the final dataset
    final_text = "\n\n".join(combined_segments)

    # Tokenize using GPT-2 standard tokenizer
    logger.info("Tokenizing with GPT-2 tiktoken...")
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(final_text)

    # Create a small validation split (e.g., last 10%)
    n = len(train_ids)
    val_idx = int(n * 0.9)
    train_data = train_ids[:val_idx]
    val_data = train_ids[val_idx:]

    logger.info(f"Train has {len(train_data):,} tokens")
    logger.info(f"Validation has {len(val_data):,} tokens")

    # Save to binary files
    logger.info("Saving to train.bin and val.bin...")
    train_arr = np.array(train_data, dtype=np.uint16)
    val_arr = np.array(val_data, dtype=np.uint16)

    train_arr.tofile(TRAIN_BIN_PATH)
    val_arr.tofile(VAL_BIN_PATH)

    logger.info("Pipeline preparation complete.")
