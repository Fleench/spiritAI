"""Shared SpiritAI checkpoint loading and text generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
import threading

import tiktoken
import torch

from model import GPTConfig, GPTLanguageModel
from paths import workspace_path

GPT2_VOCAB_SIZE = 50_257
DEFAULT_OUTPUT_DIR = workspace_path("models")


@dataclass(frozen=True)
class GenerationSettings:
    """Sampling settings used when generating from the local checkpoint."""

    max_new_tokens: int = 160
    temperature: float = 0.8
    top_k: int = 50
    repetition_penalty: float = 1.15

    @classmethod
    def from_env(cls) -> "GenerationSettings":
        return cls(
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", str(cls.max_new_tokens))),
            temperature=float(os.getenv("TEMPERATURE", str(cls.temperature))),
            top_k=int(os.getenv("TOP_K", str(cls.top_k))),
            repetition_penalty=float(
                os.getenv("REPETITION_PENALTY", str(cls.repetition_penalty))
            ),
        )


@dataclass(frozen=True)
class ModelInfo:
    """Human-readable information about a loaded SpiritAI checkpoint."""

    device: str
    n_embd: int
    n_head: int
    n_layer: int
    block_size: int
    vocab_size: int


class SpiritGenerator:
    """Lazy reusable wrapper around the trained SpiritAI Nano-GPT checkpoint."""

    def __init__(self, settings: GenerationSettings | None = None) -> None:
        self.settings = settings or GenerationSettings.from_env()
        self.output_dir = DEFAULT_OUTPUT_DIR
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = tiktoken.get_encoding("gpt2")
        self.config, self.model = self._load_model()
        self._lock = threading.Lock()

    def _load_model(self) -> tuple[GPTConfig, GPTLanguageModel]:
        config_path = self.output_dir / "config.json"
        model_path = self.output_dir / "nano_gpt_model.pt"

        if not config_path.exists():
            raise FileNotFoundError(
                f"config.json not found at {config_path}; run nano_gpt.py first"
            )
        if not model_path.exists():
            raise FileNotFoundError(
                f"nano_gpt_model.pt not found at {model_path}; run nano_gpt.py first"
            )

        config = GPTConfig.from_json(config_path)
        config.vocab_size = GPT2_VOCAB_SIZE
        model = GPTLanguageModel(config)
        model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        model.to(self.device)
        model.eval()
        return config, model

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            device=self.device,
            n_embd=self.config.n_embd,
            n_head=self.config.n_head,
            n_layer=self.config.n_layer,
            block_size=self.config.block_size,
            vocab_size=self.config.vocab_size,
        )

    def encode(self, text: str) -> list[int]:
        return self.encoder.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.encoder.decode([int(i) for i in ids])

    def generate(self, prompt: str) -> str:
        """Return decoded model output for a non-empty prompt."""
        context_idx = self.encode(prompt)
        if not context_idx:
            return "[No recognizable tokens in prompt.]"

        x = torch.tensor(
            [context_idx[-self.config.block_size :]],
            dtype=torch.long,
            device=self.device,
        )
        with self._lock, torch.no_grad():
            y = self.model.generate(
                x,
                max_new_tokens=self.settings.max_new_tokens,
                temperature=self.settings.temperature,
                top_k=self.settings.top_k,
                repetition_penalty=self.settings.repetition_penalty,
            )
        return self.decode(y[0].tolist())


def trim_prompt(generated_text: str, prompt: str) -> str:
    """Remove the echoed prompt from a generation when it is present."""
    if generated_text.startswith(prompt):
        completion = generated_text[len(prompt) :].strip()
        return completion or generated_text.strip()
    return generated_text.strip()
