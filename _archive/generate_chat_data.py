"""Generate JSONL question/answer chat data from the cleaned SpiritAI corpus.

The script uses Gemini to turn sentence-aware, overlapping chunks of ``clean.txt``
into prompt/response pairs suitable for supervised chat fine-tuning data.

Usage:
    python generate_chat_data.py
    python generate_chat_data.py --input /workspace/data/clean.txt --output /workspace/data/chat_data.jsonl

Configuration:
    Put GEMINI_API_KEY=your_key_here in .env (or export it in the shell).
    Set --concurrent-ai-calls, or CONCURRENT_AI_CALLS, to control parallel calls.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import re
import threading
import time
from typing import Any

from dotenv import load_dotenv
from paths import workspace_path

DEFAULT_INPUT_FILE = workspace_path("data", "clean.txt")
DEFAULT_OUTPUT_FILE = workspace_path("data", "chat_data.jsonl")
DEFAULT_CHUNK_SIZE = 1_500
DEFAULT_CHUNK_OVERLAP = 300
DEFAULT_MIN_CHUNK_CHARS = 200
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_SLEEP_SECONDS = 3.0
DEFAULT_RATE_LIMIT_BACKOFF_SECONDS = 10.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_CONCURRENT_AI_CALLS = int(os.getenv("CONCURRENT_AI_CALLS", "5"))
DEFAULT_DEDUPE_THRESHOLD = 0.9
MIN_PROMPT_CHARS = 10
MAX_PROMPT_CHARS = 200
MIN_RESPONSE_CHARS = 50
MAX_RESPONSE_CHARS = 2_000
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class TextChunk:
    """A chunk of source text plus stable metadata for resume tracking."""

    number: int
    start: int
    end: int
    text: str

    @property
    def chunk_id(self) -> str:
        return f"{self.number:06d}-{self.start}-{self.end}"


@dataclass
class GenerationStats:
    """Counters and reason lists used for the final progress summary."""

    total_chunks: int = 0
    processed_chunks: int = 0
    resumed_chunks: int = 0
    skipped_chunks: list[str] = field(default_factory=list)
    failed_chunks: list[str] = field(default_factory=list)
    successful_pairs: int = 0
    duplicate_pairs: int = 0
    invalid_pairs: int = 0


@dataclass
class RateLimitState:
    """Mutable delay state for adaptive rate-limit handling."""

    delay_seconds: float
    minimum_delay_seconds: float
    rate_limit_backoff_seconds: float
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def current_delay(self) -> float:
        with self.lock:
            return self.delay_seconds

    def slow_down(self) -> float:
        with self.lock:
            self.delay_seconds = max(self.delay_seconds * 2, self.rate_limit_backoff_seconds)
            return self.delay_seconds

    def speed_up(self) -> float:
        with self.lock:
            if self.delay_seconds > self.minimum_delay_seconds:
                self.delay_seconds = max(self.minimum_delay_seconds, self.delay_seconds * 0.8)
            return self.delay_seconds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SpiritAI chat Q&A JSONL data from cleaned theological text."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE, help="Clean corpus text file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Destination JSONL file. New pairs are appended by default.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Resume-state JSON file. Defaults to <output>.progress.json.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Structured log file. Defaults to <output>.log.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Target number of characters to send to Gemini in each request.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Characters of context to overlap between chunks.",
    )
    parser.add_argument(
        "--min-chunk-chars",
        type=int,
        default=DEFAULT_MIN_CHUNK_CHARS,
        help="Skip chunks shorter than this many characters.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name to use.")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Initial delay between Gemini calls to reduce rate-limit pressure.",
    )
    parser.add_argument(
        "--rate-limit-backoff-seconds",
        type=float,
        default=DEFAULT_RATE_LIMIT_BACKOFF_SECONDS,
        help="Minimum retry delay after Gemini returns a rate-limit error.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Retry each failed chunk up to this many times with exponential backoff.",
    )
    parser.add_argument(
        "--concurrent-ai-calls",
        type=int,
        default=DEFAULT_CONCURRENT_AI_CALLS,
        help=(
            "Maximum number of Gemini chunks to process at the same time. "
            "Defaults to CONCURRENT_AI_CALLS or 5."
        ),
    )
    parser.add_argument(
        "--dedupe-threshold",
        type=float,
        default=DEFAULT_DEDUPE_THRESHOLD,
        help="Prompt similarity threshold for duplicate detection. Use 1.0 for exact-only dedupe.",
    )
    parser.add_argument(
        "--disable-dedupe",
        action="store_true",
        help="Disable duplicate and near-duplicate prompt filtering.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output and resume-state files instead of appending/resuming.",
    )
    return parser.parse_args()


def configure_logging(log_file: Path) -> logging.Logger:
    """Create a logger that writes timestamped chunk events to a file and stderr."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("generate_chat_data")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    formatter.converter = time.gmtime

    file_handler = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def configure_gemini(model_name: str) -> Any:
    """Load .env, configure Gemini, and return the selected model."""
    import google.generativeai as genai

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to .env or export it before running.")

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def strip_markdown_fence(text: str) -> str:
    """Remove accidental Markdown fences while preserving the JSON payload."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def validate_qa_pairs(value: Any) -> tuple[list[dict[str, str]], list[str]]:
    """Return valid prompt/response dicts plus reasons for discarded objects."""
    if not isinstance(value, list):
        raise ValueError("Gemini response was not a JSON array")

    pairs: list[dict[str, str]] = []
    rejected: list[str] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            rejected.append(f"item {index} is not an object")
            continue

        prompt = item.get("prompt")
        response = item.get("response")
        if not isinstance(prompt, str) or not isinstance(response, str):
            rejected.append(f"item {index} prompt/response fields must be strings")
            continue

        prompt = prompt.strip()
        response = response.strip()
        if not prompt or not response:
            rejected.append(f"item {index} prompt/response fields cannot be blank")
            continue
        if not MIN_PROMPT_CHARS <= len(prompt) <= MAX_PROMPT_CHARS:
            rejected.append(
                f"item {index} prompt length {len(prompt)} outside {MIN_PROMPT_CHARS}-{MAX_PROMPT_CHARS}"
            )
            continue
        if not MIN_RESPONSE_CHARS <= len(response) <= MAX_RESPONSE_CHARS:
            rejected.append(
                f"item {index} response length {len(response)} "
                f"outside {MIN_RESPONSE_CHARS}-{MAX_RESPONSE_CHARS}"
            )
            continue

        pairs.append({"prompt": prompt, "response": response})
    return pairs, rejected


def generate_qa_for_chunk(model: Any, text_chunk: str) -> tuple[list[dict[str, str]], list[str]]:
    prompt = f"""
Read the following theological text and generate 3 to 5 high-quality Question/Answer pairs
based strictly on the content.

Guidelines:
- The "prompt" should be a question a curious user might ask.
- The "prompt" must be {MIN_PROMPT_CHARS} to {MAX_PROMPT_CHARS} characters long.
- The "response" should be accurate to the text, written in the tone of a wise, helpful assistant.
- The "response" must be {MIN_RESPONSE_CHARS} to {MAX_RESPONSE_CHARS} characters long.
- Neither field may be empty or whitespace-only.
- Do not mention "the text says" in the response; answer directly.

Respond ONLY with a valid JSON array. Do not use markdown blocks. Example format:
[
  {{"prompt": "What is the nature of the spirit?", "response": "The spirit is eternal..."}}
]

Text to analyze:
{text_chunk}
"""

    response = model.generate_content(prompt)
    text = strip_markdown_fence(response.text)
    return validate_qa_pairs(json.loads(text))


def find_sentence_boundary(text: str, lower_bound: int, upper_bound: int) -> int:
    """Find a sentence boundary near the desired chunk end without splitting mid-sentence."""
    candidate = upper_bound
    for match in SENTENCE_BOUNDARY_RE.finditer(text, lower_bound, upper_bound):
        candidate = match.end()
    return candidate


def find_overlap_start(text: str, current_start: int, desired_start: int, end: int) -> int:
    """Start the overlapped chunk on the nearest sentence boundary when possible."""
    boundary_candidates = [
        match.end()
        for match in SENTENCE_BOUNDARY_RE.finditer(text, current_start + 1, end)
        if match.end() <= desired_start
    ]
    if boundary_candidates:
        return boundary_candidates[-1]
    return desired_start


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[TextChunk]:
    """Split text into sentence-aware chunks with overlap for context continuity."""
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than zero")
    if overlap < 0:
        raise ValueError("--chunk-overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("--chunk-overlap must be smaller than --chunk-size")

    chunks: list[TextChunk] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        target_end = min(start + chunk_size, text_length)
        if target_end < text_length:
            lower_bound = min(start + max(chunk_size // 2, 1), target_end)
            end = find_sentence_boundary(text, lower_bound, target_end)
        else:
            end = target_end

        chunk_body = text[start:end].strip()
        if chunk_body:
            chunks.append(TextChunk(number=len(chunks) + 1, start=start, end=end, text=chunk_body))

        if end >= text_length:
            break
        desired_start = max(0, end - overlap)
        next_start = find_overlap_start(text, start, desired_start, end)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def load_resume_state(state_file: Path, overwrite: bool) -> dict[str, Any]:
    """Load chunk status from disk, optionally resetting it for a fresh run."""
    if overwrite or not state_file.exists():
        return {"chunks": {}}
    with state_file.open(encoding="utf-8") as f:
        state = json.load(f)
    if not isinstance(state, dict) or not isinstance(state.get("chunks"), dict):
        raise ValueError(f"Resume state file is malformed: {state_file}")
    return state


def save_resume_state(state_file: Path, state: dict[str, Any]) -> None:
    """Persist resume metadata after every chunk so interrupted runs can resume."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = state_file.with_suffix(state_file.suffix + ".tmp")
    with tmp_file.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp_file.replace(state_file)


def normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.casefold()).strip()


def load_existing_prompts(output_file: Path, overwrite: bool) -> set[str]:
    """Seed deduplication from existing JSONL output when resuming."""
    if overwrite or not output_file.exists():
        return set()

    prompts: set[str] = set()
    with output_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = item.get("prompt") if isinstance(item, dict) else None
            if isinstance(prompt, str) and prompt.strip():
                prompts.add(normalize_prompt(prompt))
    return prompts


def is_duplicate_prompt(prompt: str, seen_prompts: set[str], threshold: float) -> bool:
    """Detect exact or near-duplicate prompts using normalized similarity."""
    normalized = normalize_prompt(prompt)
    if normalized in seen_prompts:
        return True
    if threshold >= 1.0:
        return False
    return any(SequenceMatcher(None, normalized, seen).ratio() >= threshold for seen in seen_prompts)


@dataclass(frozen=True)
class ChunkGenerationResult:
    """Generated Q&A and validation metadata for one source chunk."""

    chunk: TextChunk
    qa_pairs: list[dict[str, str]]
    rejected: list[str]
    failure_reason: str | None


def process_chunk(
    model: Any,
    chunk: TextChunk,
    max_retries: int,
    rate_limit: RateLimitState,
    logger: logging.Logger,
) -> ChunkGenerationResult:
    """Generate Q&A for one chunk and throttle this worker before it takes more work."""
    qa_pairs, rejected, failure_reason = generate_with_retries(
        model=model,
        chunk=chunk,
        max_retries=max_retries,
        rate_limit=rate_limit,
        logger=logger,
    )
    delay = rate_limit.current_delay()
    if delay > 0:
        time.sleep(delay)
    return ChunkGenerationResult(
        chunk=chunk,
        qa_pairs=qa_pairs,
        rejected=rejected,
        failure_reason=failure_reason,
    )


def generate_with_retries(
    model: Any,
    chunk: TextChunk,
    max_retries: int,
    rate_limit: RateLimitState,
    logger: logging.Logger,
) -> tuple[list[dict[str, str]], list[str], str | None]:
    """Generate Q&A for a chunk, retrying transient errors with exponential backoff."""
    from google.api_core.exceptions import ResourceExhausted

    attempts = max_retries + 1
    delay = max(1.0, rate_limit.current_delay())
    for attempt in range(1, attempts + 1):
        try:
            qa_pairs, rejected = generate_qa_for_chunk(model, chunk.text)
            rate_limit.speed_up()
            return qa_pairs, rejected, None
        except ResourceExhausted as exc:
            if attempt == attempts:
                return [], [], f"rate limited after {attempt} attempts: {exc}"
            delay = max(delay, rate_limit.slow_down())
            logger.warning(
                "chunk=%s attempt=%s/%s status=rate_limited retry_in=%.2fs error=%s",
                chunk.number,
                attempt,
                attempts,
                delay,
                exc,
            )
            time.sleep(delay)
            delay *= 2
        except Exception as exc:
            if attempt == attempts:
                return [], [], f"failed after {attempt} attempts: {exc}"
            logger.warning(
                "chunk=%s attempt=%s/%s status=retry retry_in=%.2fs error=%s",
                chunk.number,
                attempt,
                attempts,
                delay,
                exc,
            )
            time.sleep(delay)
            delay *= 2
    return [], [], "retry loop exited unexpectedly"


def main() -> None:
    args = parse_args()
    input_file = args.input.expanduser()
    output_file = args.output.expanduser()
    state_file = (
        args.state_file or output_file.with_suffix(output_file.suffix + ".progress.json")
    ).expanduser()
    log_file = (args.log_file or output_file.with_suffix(output_file.suffix + ".log")).expanduser()
    logger = configure_logging(log_file)

    if not input_file.exists():
        raise FileNotFoundError(f"Could not find {input_file}")
    if args.max_retries < 0:
        raise ValueError("--max-retries cannot be negative")
    if args.sleep_seconds < 0:
        raise ValueError("--sleep-seconds cannot be negative")
    if args.rate_limit_backoff_seconds < 0:
        raise ValueError("--rate-limit-backoff-seconds cannot be negative")
    if args.concurrent_ai_calls < 1:
        raise ValueError("--concurrent-ai-calls must be at least 1")
    if not 0 <= args.dedupe_threshold <= 1:
        raise ValueError("--dedupe-threshold must be between 0 and 1")

    logger.info(
        "event=start input=%s output=%s state=%s log=%s concurrent_ai_calls=%s",
        input_file,
        output_file,
        state_file,
        log_file,
        args.concurrent_ai_calls,
    )
    model = configure_gemini(args.model)

    logger.info("event=read_corpus path=%s", input_file)
    corpus = input_file.read_text(encoding="utf-8")
    chunks = chunk_text(corpus, args.chunk_size, args.chunk_overlap)
    stats = GenerationStats(total_chunks=len(chunks))
    logger.info(
        "event=chunked total_chunks=%s chunk_size=%s overlap=%s min_chunk_chars=%s",
        len(chunks),
        args.chunk_size,
        args.chunk_overlap,
        args.min_chunk_chars,
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        output_file.write_text("", encoding="utf-8")
        if state_file.exists():
            state_file.unlink()

    state = load_resume_state(state_file, args.overwrite)
    chunk_state: dict[str, Any] = state.setdefault("chunks", {})
    seen_prompts = load_existing_prompts(output_file, args.overwrite)
    rate_limit = RateLimitState(
        delay_seconds=args.sleep_seconds,
        minimum_delay_seconds=args.sleep_seconds,
        rate_limit_backoff_seconds=args.rate_limit_backoff_seconds,
    )

    pending_chunks: list[TextChunk] = []
    for chunk in chunks:
        existing = chunk_state.get(chunk.chunk_id, {})
        if existing.get("status") in {"success", "skipped"}:
            stats.resumed_chunks += 1
            logger.info(
                "chunk=%s/%s status=resumed previous_status=%s",
                chunk.number,
                len(chunks),
                existing.get("status"),
            )
            continue

        if len(chunk.text) < args.min_chunk_chars:
            reason = f"chunk {chunk.number}: below min size ({len(chunk.text)} chars)"
            stats.skipped_chunks.append(reason)
            chunk_state[chunk.chunk_id] = {"status": "skipped", "reason": reason}
            save_resume_state(state_file, state)
            logger.info("chunk=%s/%s status=skipped reason=%s", chunk.number, len(chunks), reason)
            continue

        pending_chunks.append(chunk)

    logger.info(
        "event=dispatch pending_chunks=%s concurrent_ai_calls=%s",
        len(pending_chunks),
        args.concurrent_ai_calls,
    )

    def record_result(result: ChunkGenerationResult) -> None:
        chunk = result.chunk
        stats.invalid_pairs += len(result.rejected)

        if result.failure_reason:
            reason = f"chunk {chunk.number}: {result.failure_reason}"
            stats.failed_chunks.append(reason)
            chunk_state[chunk.chunk_id] = {
                "status": "failed",
                "reason": reason,
                "invalid_pairs": result.rejected,
            }
            save_resume_state(state_file, state)
            logger.error("chunk=%s/%s status=failed reason=%s", chunk.number, len(chunks), result.failure_reason)
            return

        written_pairs = 0
        duplicate_pairs = 0
        for pair in result.qa_pairs:
            if not args.disable_dedupe and is_duplicate_prompt(
                pair["prompt"], seen_prompts, args.dedupe_threshold
            ):
                duplicate_pairs += 1
                continue
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            seen_prompts.add(normalize_prompt(pair["prompt"]))
            written_pairs += 1
        f.flush()

        stats.processed_chunks += 1
        stats.successful_pairs += written_pairs
        stats.duplicate_pairs += duplicate_pairs
        chunk_state[chunk.chunk_id] = {
            "status": "success",
            "pairs_written": written_pairs,
            "duplicates_skipped": duplicate_pairs,
            "invalid_pairs": result.rejected,
        }
        save_resume_state(state_file, state)
        logger.info(
            "chunk=%s/%s status=success pairs_written=%s duplicates=%s invalid=%s next_delay=%.2fs",
            chunk.number,
            len(chunks),
            written_pairs,
            duplicate_pairs,
            len(result.rejected),
            rate_limit.current_delay(),
        )

    with output_file.open("a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.concurrent_ai_calls) as executor:
            future_to_chunk = {}
            for chunk in pending_chunks:
                logger.info("chunk=%s/%s status=queued chars=%s", chunk.number, len(chunks), len(chunk.text))
                future = executor.submit(
                    process_chunk,
                    model,
                    chunk,
                    args.max_retries,
                    rate_limit,
                    logger,
                )
                future_to_chunk[future] = chunk

            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    record_result(future.result())
                except Exception as exc:
                    reason = f"chunk {chunk.number}: worker failed unexpectedly: {exc}"
                    stats.failed_chunks.append(reason)
                    chunk_state[chunk.chunk_id] = {"status": "failed", "reason": reason}
                    save_resume_state(state_file, state)
                    logger.exception("chunk=%s/%s status=failed reason=%s", chunk.number, len(chunks), reason)

    print("\nProgress summary")
    print("================")
    print(f"Total chunks: {stats.total_chunks}")
    print(f"Chunks processed this run: {stats.processed_chunks}")
    print(f"Chunks skipped from resume state: {stats.resumed_chunks}")
    print(f"Successful pairs generated: {stats.successful_pairs}")
    print(f"Duplicate pairs skipped: {stats.duplicate_pairs}")
    print(f"Invalid pairs rejected: {stats.invalid_pairs}")
    print(f"Skipped chunks: {len(stats.skipped_chunks)}")
    for reason in stats.skipped_chunks:
        print(f"  - {reason}")
    print(f"Failed chunks: {len(stats.failed_chunks)}")
    for reason in stats.failed_chunks:
        print(f"  - {reason}")
    print(f"Output file: {output_file}")
    print(f"Resume state: {state_file}")
    print(f"Log file: {log_file}")


if __name__ == "__main__":
    main()
