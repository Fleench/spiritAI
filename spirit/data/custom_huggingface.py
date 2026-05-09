"""Custom Hugging Face dataset configuration helpers.

Users can add arbitrary Hugging Face datasets to
``spirit/data/huggingface_datasets.json`` without changing Python code.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spirit.config import HUGGINGFACE_DATASETS_CONFIG_PATH

SUPPORTED_DATA_TYPES = {"auto", "qa", "text"}


@dataclass(frozen=True)
class HuggingFaceDatasetConfig:
    """Normalized configuration for one custom Hugging Face dataset."""

    path: str
    split: str = "train"
    config: str | None = None
    data_type: str = "auto"
    prompt_columns: tuple[str, ...] = ()
    response_columns: tuple[str, ...] = ()
    text_columns: tuple[str, ...] = ()
    output_file: str = ""
    max_rows: int | None = None
    streaming: bool = False
    weight: int = 1


def slugify_dataset_path(dataset_path: str) -> str:
    """Create a stable filename stem from a Hugging Face dataset path."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", dataset_path).strip("_").lower()
    return slug or "huggingface_dataset"


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(item) for item in value if str(item).strip())
    raise ValueError(f"Expected a string or list of strings, got {type(value).__name__}")


def _as_optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{field_name} must be greater than zero")
    return parsed


def _normalize_entry(entry: dict[str, Any]) -> HuggingFaceDatasetConfig:
    path = str(entry.get("path") or entry.get("name") or "").strip()
    if not path:
        raise ValueError("Every custom Hugging Face dataset needs a non-empty 'path'.")

    data_type = str(entry.get("type") or entry.get("data_type") or "auto").strip().lower()
    if data_type not in SUPPORTED_DATA_TYPES:
        allowed = ", ".join(sorted(SUPPORTED_DATA_TYPES))
        raise ValueError(f"Unsupported type '{data_type}' for {path}; expected one of: {allowed}.")

    split = str(entry.get("split") or "train").strip() or "train"
    config = entry.get("config") or entry.get("configuration")
    config = str(config).strip() if config is not None and str(config).strip() else None

    output_file = str(entry.get("output_file") or f"{slugify_dataset_path(path)}.txt").strip()
    if not output_file.endswith(".txt"):
        output_file = f"{output_file}.txt"
    if Path(output_file).name != output_file:
        raise ValueError("output_file must be a filename, not a path.")

    return HuggingFaceDatasetConfig(
        path=path,
        split=split,
        config=config,
        data_type=data_type,
        prompt_columns=_as_tuple(entry.get("prompt_columns") or entry.get("question_columns")),
        response_columns=_as_tuple(entry.get("response_columns") or entry.get("answer_columns")),
        text_columns=_as_tuple(entry.get("text_columns")),
        output_file=output_file,
        max_rows=_as_optional_int(entry.get("max_rows"), "max_rows"),
        streaming=bool(entry.get("streaming", False)),
        weight=max(1, int(entry.get("weight", 1))),
    )


def load_huggingface_dataset_configs(
    config_path: str | Path = HUGGINGFACE_DATASETS_CONFIG_PATH,
) -> list[HuggingFaceDatasetConfig]:
    """Load custom Hugging Face dataset settings from a JSON file.

    The file can be either a raw list of dataset objects or an object with a
    ``datasets`` list. Missing files simply mean no custom datasets are enabled.
    """
    path = Path(config_path)
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("datasets", payload) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("Custom Hugging Face config must be a list or contain a 'datasets' list.")

    return [_normalize_entry(entry) for entry in entries if isinstance(entry, dict)]
