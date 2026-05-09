"""Convert chat JSONL prompt/response pairs into a plain-text chat corpus.

The generated corpus uses stable, all-caps turn markers so the tokenizer can
learn distinct chat boundaries before Nano-GPT training.

Usage:
    python format_chat_corpus.py
    python format_chat_corpus.py --input /workspace/data/chat_data.jsonl --output /workspace/raw_data/chat_corpus.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paths import workspace_path

DEFAULT_INPUT_FILE = workspace_path("data", "chat_data.jsonl")
DEFAULT_OUTPUT_FILE = workspace_path("raw_data", "chat_corpus.txt")
USER_MARKER = "USER_PROMPT"
AI_MARKER = "AI_RESPONSE"
TURN_END_MARKER = "END_OF_TURN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Format chat_data.jsonl prompt/response pairs as a plain-text chat corpus."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Source JSONL file containing prompt and response fields.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Destination text corpus path.",
    )
    return parser.parse_args()


def clean_field(value: Any, field_name: str, line_number: int) -> str:
    """Return a single-line string for a required JSONL field."""
    if value is None:
        raise ValueError(f"Line {line_number} is missing required field: {field_name}")

    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise ValueError(f"Line {line_number} has an empty {field_name!r} field")
    return text


def format_chat_corpus(input_path: Path) -> str:
    """Read chat JSONL and return text with explicit user/AI turn markers."""
    formatted_turns: list[str] = []

    with input_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_number} is not valid JSON: {exc.msg}") from exc

            prompt = clean_field(data.get("prompt"), "prompt", line_number)
            response = clean_field(data.get("response"), "response", line_number)
            formatted_turns.append(
                f"{USER_MARKER} {prompt}\n"
                f"{AI_MARKER} {response} {TURN_END_MARKER}"
            )

    return "\n\n".join(formatted_turns) + ("\n" if formatted_turns else "")


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input JSONL file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_chat_corpus(args.input), encoding="utf-8")
    print(f"Chat corpus created at {args.output}!")


if __name__ == "__main__":
    main()
