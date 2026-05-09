"""Data source fetchers for building the SpiritAI corpus.

This module downloads static texts and HuggingFace datasets, formatting
them appropriately to be sanitized and processed by the pipeline.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, TypeVar

import requests
from datasets import load_dataset # type: ignore

from spirit.config import CUSTOM_HUGGINGFACE_RAW_DIR, RAW_DATA_DIR, ROOT_DIR
from spirit.data.custom_huggingface import HuggingFaceDatasetConfig, load_huggingface_dataset_configs
from spirit.data.format import format_chat_turn

logger = logging.getLogger(__name__)

# Target topics for Wikipedia filtering
WIKI_KEYWORDS = {"religion", "philosophy", "christianity", "theology", "god", "church", "bible"}

T = TypeVar("T")

QA_PROMPT_CANDIDATES = ("question", "prompt", "instruction", "query", "user", "input", "title")
QA_RESPONSE_CANDIDATES = ("answer", "response", "output", "completion", "assistant", "text")
TEXT_CANDIDATES = ("text", "content", "article", "body", "passage", "answer", "response", "output")


def _format_failure(source_name: str, exc: Exception) -> str:
    """Return a concise, user-facing download failure message."""
    message = str(exc).strip()
    if message:
        return f"{source_name} ({message})"
    return source_name


def _fetch_and_save(
    source_name: str,
    download_fn: Callable[[], T],
    save_fn: Callable[[T], None],
    failures: list[str],
) -> None:
    """Fetch one source and record failures without stopping the download run."""
    try:
        data = download_fn()
        save_fn(data)
    except Exception as exc:
        failures.append(_format_failure(source_name, exc))
        logger.debug("Could not download %s", source_name, exc_info=True)


def _write_text(filename: str, text: str) -> None:
    """Write raw text to the raw data directory."""
    with open(RAW_DATA_DIR / filename, "w", encoding="utf-8") as f:
        f.write(text)


def _write_turns(filename: str, turns: list[str]) -> None:
    """Write formatted chat turns to the raw data directory."""
    _write_text(filename, "\n\n".join(turns))


def _write_path(path: Path, text: str) -> None:
    """Write text to an explicit path, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def download_quora_question_answer() -> list[str]:
    """Download and format toughdata/quora-question-answer-dataset JSON rows."""
    logger.info("Downloading toughdata/quora-question-answer-dataset...")
    dataset = load_dataset("toughdata/quora-question-answer-dataset", split="train")

    turns = []
    for row in dataset:
        question = str(row["question"]).strip() # type: ignore
        answer = str(row["answer"]).strip() # type: ignore
        if question and answer:
            turns.append(format_chat_turn(question, answer))

    return turns


def _first_present_text(row: Mapping[str, Any], columns: Iterable[str]) -> str:
    """Return the first non-empty string from the requested row columns."""
    for column in columns:
        if column in row and row[column] is not None:
            text = str(row[column]).strip()
            if text:
                return text
    return ""


def _combine_present_text(row: Mapping[str, Any], columns: Iterable[str]) -> str:
    """Join all non-empty strings from the requested row columns."""
    parts = []
    for column in columns:
        if column in row and row[column] is not None:
            text = str(row[column]).strip()
            if text:
                parts.append(text)
    return "\n".join(parts)


def _detect_qa_columns(row: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """Infer prompt/response columns from common instruction and Q&A schemas."""
    keys = {key.lower(): key for key in row.keys()}

    prompt_columns = [keys[name] for name in QA_PROMPT_CANDIDATES if name in keys]
    response_columns = [keys[name] for name in QA_RESPONSE_CANDIDATES if name in keys]

    # Alpaca-style rows often use instruction + optional input as the prompt.
    if "instruction" in keys and "input" in keys:
        prompt_columns = [keys["instruction"], keys["input"]]

    if prompt_columns and response_columns:
        # Do not use the same column as both sides of a chat turn.
        prompt_tuple = tuple(prompt_columns)
        response_tuple = tuple(col for col in response_columns if col not in prompt_tuple)
        if response_tuple:
            return prompt_tuple, response_tuple

    return None


def _detect_text_columns(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Infer plain-text columns from common corpus schemas."""
    keys = {key.lower(): key for key in row.keys()}
    for candidate in TEXT_CANDIDATES:
        if candidate in keys and str(row[keys[candidate]]).strip():
            return (keys[candidate],)

    string_columns = [key for key, value in row.items() if isinstance(value, str) and value.strip()]
    return tuple(string_columns[:1])


def _format_custom_huggingface_row(
    row: Mapping[str, Any],
    source: HuggingFaceDatasetConfig,
) -> str:
    """Format one arbitrary Hugging Face row as chat or plain text."""
    prompt_columns = source.prompt_columns
    response_columns = source.response_columns
    text_columns = source.text_columns
    data_type = source.data_type

    if data_type in {"auto", "qa"}:
        if not prompt_columns or not response_columns:
            detected = _detect_qa_columns(row)
            if detected:
                prompt_columns = prompt_columns or detected[0]
                response_columns = response_columns or detected[1]

        if prompt_columns and response_columns:
            prompt = _combine_present_text(row, prompt_columns)
            response = _first_present_text(row, response_columns)
            if prompt and response:
                return format_chat_turn(prompt, response)

        if data_type == "qa":
            return ""

    if not text_columns:
        text_columns = _detect_text_columns(row)

    return _combine_present_text(row, text_columns)


def download_custom_huggingface_dataset(source: HuggingFaceDatasetConfig) -> list[str]:
    """Download and format a configured Hugging Face dataset."""
    logger.info("Downloading custom Hugging Face dataset %s...", source.path)
    load_kwargs: dict[str, Any] = {"split": source.split}
    if source.config:
        load_kwargs["name"] = source.config
    if source.streaming:
        load_kwargs["streaming"] = True

    dataset = load_dataset(source.path, **load_kwargs)
    rows = []
    for index, row in enumerate(dataset):
        if source.max_rows is not None and index >= source.max_rows:
            break
        if not isinstance(row, Mapping):
            continue
        formatted = _format_custom_huggingface_row(row, source)
        if formatted:
            rows.append(formatted)

    return rows


def fetch_custom_huggingface_sources(config_path: str | Path | None = None) -> list[str]:
    """Fetch user-configured Hugging Face datasets and save them as raw text."""
    failures: list[str] = []
    sources = load_huggingface_dataset_configs(config_path) if config_path else load_huggingface_dataset_configs()
    if not sources:
        logger.info("No custom Hugging Face datasets configured.")
        return failures

    for source in sources:
        _fetch_and_save(
            source.path,
            lambda source=source: download_custom_huggingface_dataset(source),
            lambda rows, source=source: _write_path(
                CUSTOM_HUGGINGFACE_RAW_DIR / source.output_file, "\n\n".join(rows)
            ),
            failures,
        )

    return failures

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


def fetch_all_sources(config_path: str | Path | None = None) -> list[str]:
    """Fetch all configured data sources and save them to the raw data directory.

    Returns:
        A list of user-facing names for sources that could not be downloaded.
    """
    logger.info("Starting data source downloads...")
    failures: list[str] = []

    _fetch_and_save(
        "Ante-Nicene Fathers",
        download_ante_nicene_fathers,
        lambda text: _write_text("ante_nicene_fathers.txt", text),
        failures,
    )
    _fetch_and_save(
        "CPDV Bible",
        load_cpdv_bible,
        lambda text: _write_text("bible.txt", text),
        failures,
    )
    _fetch_and_save(
        "yahma/alpaca-cleaned",
        download_alpaca,
        lambda turns: _write_turns("alpaca.txt", turns),
        failures,
    )
    _fetch_and_save(
        "Malalatiana/theological-questions-answers",
        download_theological_qa,
        lambda turns: _write_turns("theological_qa.txt", turns),
        failures,
    )
    _fetch_and_save(
        "toughdata/quora-question-answer-dataset",
        download_quora_question_answer,
        lambda turns: _write_turns("quora_question_answer.txt", turns),
        failures,
    )
    _fetch_and_save(
        "wikimedia/wikipedia",
        download_wikipedia,
        lambda articles: _write_text("wikipedia.txt", "\n\n".join(articles)),
        failures,
    )

    failures.extend(fetch_custom_huggingface_sources(config_path))

    if failures:
        logger.warning("Could not download: %s", "; ".join(failures))
    else:
        logger.info("Finished downloading all data sources.")

    return failures
