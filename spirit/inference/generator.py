"""Inference generator for the SpiritAI model.

Provides a unified interface for loading a model checkpoint, formatting user
prompts, and generating responses with proper stop-sequence handling.
"""

from __future__ import annotations

import logging

import tiktoken
import torch

from spirit.config import CHECKPOINT_PATH, GenerateConfig, ModelConfig
from spirit.data.format import END_OF_TURN, format_prompt
from spirit.model import GPT

logger = logging.getLogger(__name__)


class SpiritGenerator:
    """Wrapper class for generating responses from the NanoGPT model."""

    def __init__(self) -> None:
        """Initialize the generator and load the model."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = GenerateConfig()
        self.enc = tiktoken.get_encoding("gpt2")

        # Load model
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(f"No checkpoint found at {CHECKPOINT_PATH}. Train the model first.")

        logger.info(f"Loading model from {CHECKPOINT_PATH} to {self.device}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)

        # Reconstruct model config from checkpoint
        self.model_config = ModelConfig(**checkpoint['model_args'])

        # Initialize and load weights
        self.model = GPT(self.model_config)
        state_dict = checkpoint['model']

        # Strip DDP/compile prefix if present
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

        # The GPT-2 token ID for END_OF_TURN isn't a single token, but we can match sequences.
        # Alternatively, we could yield token by token and stop when we decode END_OF_TURN.
        # For simplicity and efficiency, let's decode incrementally or pass a custom stop token if END_OF_TURN mapped to one.
        # Since it doesn't, we will generate tokens one by one and check the buffer.

    @torch.no_grad()
    def generate(self, prompt: str) -> str:
        """Generate a response for the given user prompt, stopping at END_OF_TURN.

        Args:
            prompt: The user's input string.

        Returns:
            The model's generated response string.
        """
        formatted = format_prompt(prompt)
        x = torch.tensor(self.enc.encode(formatted), dtype=torch.long, device=self.device)[None, ...]

        output_tokens = []
        # Run generation token by token up to max_new_tokens
        for _ in range(self.config.max_new_tokens):
            # Crop sequence context
            idx_cond = x if x.size(1) <= self.model.config.block_size else x[:, -self.model.config.block_size:]

            # Forward pass
            logits, _ = self.model(idx_cond)
            logits = logits[:, -1, :] / self.config.temperature

            if self.config.top_k is not None:
                v, _ = torch.topk(logits, min(self.config.top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, idx_next), dim=1)
            output_tokens.append(idx_next.item())

            # Check for stop sequence
            # Decode the end of the buffer (enough to cover the stop sequence text length)
            # A rough heuristic: END_OF_TURN is 11 chars. Let's check the last 15 decoded tokens.
            buffer_text = self.enc.decode(output_tokens[-15:]) if len(output_tokens) > 15 else self.enc.decode(output_tokens)
            if END_OF_TURN in buffer_text:
                break

        # Decode full response
        out_text = self.enc.decode(output_tokens)
        if END_OF_TURN in out_text:
            out_text = out_text.split(END_OF_TURN)[0]

        return out_text.strip()
