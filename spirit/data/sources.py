"""Data source fetchers for building the SpiritAI corpus.

This module downloads static texts and HuggingFace datasets, formatting
them appropriately to be sanitized and processed by the pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import requests
from datasets import load_dataset # type: ignore

from spirit.config import RAW_DATA_DIR, ROOT_DIR
from spirit.data.format import format_chat_turn

logger = logging.getLogger(__name__)

# Target topics for Wikipedia filtering
WIKI_KEYWORDS = {"religion", "philosophy", "christianity", "theology", "god", "church", "bible"}


def download_ante_nicene_fathers() -> str:
    """Download the Ante-Nicene Fathers text."""
    logger.info("Downloading Ante-Nicene Fathers...")
    url = "https://archive.org/stream/AnteNiceneFathersCompleteVolumesIToIX_201407/Ante-nicene%20fathers%20-%20complete%20volumes%20I%20to%20IX_djvu.txt"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    response.encoding = response.encoding or "utf-8"
    return response.text


def load_cpdv_bible() -> str:
    """Load the CPDV Bible from the archived repository JSON file."""
    logger.info("Loading CPDV Bible from archive...")

    # Check archive
    archive_path = ROOT_DIR / "_archive" / "data" / "CPDV.json"
    if not archive_path.exists():
        logger.warning("CPDV Bible not found in archive. Returning empty text.")
        return ""

    try:
        with open(archive_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        texts = []
        for book in data.get("books", []):
            for chapter in book.get("chapters", []):
                for verse in chapter.get("verses", []):
                    texts.append(verse.get("text", ""))

        return "\n".join(texts)
    except Exception as e:
        logger.error(f"Error loading CPDV Bible: {e}")
        return ""


def download_alpaca() -> list[str]:
    """Download and format yahma/alpaca-cleaned dataset."""
    logger.info("Downloading yahma/alpaca-cleaned...")
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")

    turns = []
    for row in dataset:
        # Alpaca uses instruction, input, output
        instruction = str(row["instruction"]).strip() # type: ignore
        inp = str(row["input"]).strip() # type: ignore
        output = str(row["output"]).strip() # type: ignore

        # Combine instruction and input
        prompt = instruction
        if inp:
            prompt = f"{instruction}\n{inp}"

        # Only keep question-style prompts
        if "?" in prompt:
            turns.append(format_chat_turn(prompt, output))

    return turns


def download_theological_qa() -> list[str]:
    """Download and format Malalatiana/theological-questions-answers dataset."""
    logger.info("Downloading Malalatiana/theological-questions-answers...")
    dataset = load_dataset("Malalatiana/theological-questions-answers", split="train")

    turns = []
    for row in dataset:
        question = str(row["question"]).strip() # type: ignore
        answer = str(row["answer"]).strip() # type: ignore
        if question and answer:
            turns.append(format_chat_turn(question, answer))

    return turns


def download_wikipedia() -> list[str]:
    """Download and filter wikimedia/wikipedia dataset."""
    logger.info("Downloading wikimedia/wikipedia (this may take a while)...")
    # Using a smaller subset or streaming to avoid massive memory usage
    dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    articles = []
    for row in dataset:
        title = str(row["title"]).lower() # type: ignore
        text = str(row["text"]) # type: ignore

        # Fast filter on title/text keywords
        if any(keyword in title for keyword in WIKI_KEYWORDS) or \
           any(keyword in text[:500].lower() for keyword in WIKI_KEYWORDS):
            articles.append(text)
            if len(articles) >= 50000:
                break

    return articles


def fetch_all_sources() -> None:
    """Fetch all configured data sources and save them to the raw data directory."""
    logger.info("Starting data source downloads...")

    # 1. Ante-Nicene Fathers
    anf_text = download_ante_nicene_fathers()
    with open(RAW_DATA_DIR / "ante_nicene_fathers.txt", "w", encoding="utf-8") as f:
        f.write(anf_text)

    # 2. Bible
    bible_text = load_cpdv_bible()
    with open(RAW_DATA_DIR / "bible.txt", "w", encoding="utf-8") as f:
        f.write(bible_text)

    # 3. Alpaca Cleaned
    alpaca_turns = download_alpaca()
    with open(RAW_DATA_DIR / "alpaca.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(alpaca_turns))

    # 4. Theological Q&A
    theo_turns = download_theological_qa()
    with open(RAW_DATA_DIR / "theological_qa.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(theo_turns))

    # 5. Filtered Wikipedia
    wiki_articles = download_wikipedia()
    with open(RAW_DATA_DIR / "wikipedia.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(wiki_articles))

    logger.info("Finished downloading all data sources.")
