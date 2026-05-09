"""Configuration settings for SpiritAI.

This module is the single source of truth for paths, model hyper-parameters,
training configurations, and environment variable loading.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
CHECKPOINT_PATH = MODELS_DIR / "spirit_ckpt.pt"

# Create directories if they don't exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset paths
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_BIN_PATH = DATA_DIR / "train.bin"
VAL_BIN_PATH = DATA_DIR / "val.bin"

# Constants
VOCAB_SIZE = 50257  # Standard GPT-2 vocab size


@dataclass
class ModelConfig:
    """Configuration for the NanoGPT model."""
    block_size: int = 512
    vocab_size: int = VOCAB_SIZE
    n_layer: int = 16
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False


@dataclass
class TrainConfig:
    """Configuration for the training loop."""
    batch_size: int = 12
    max_iters: int = int(os.getenv("MAX_ITERS", "500000"))
    learning_rate: float = 6e-4
    grad_accum_steps: int = int(os.getenv("GRAD_ACCUM_STEPS", "4"))

    # Optimizer settings
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Learning rate decay
    decay_lr: bool = True
    warmup_iters: int = 1000
    lr_decay_iters: int = max_iters
    min_lr: float = 6e-5

    # Evaluation
    eval_interval: int = 2000
    eval_iters: int = 200
    log_interval: int = 10

    # Generation (during eval)
    always_save_checkpoint: bool = True


@dataclass
class GenerateConfig:
    """Configuration for inference generation."""
    max_new_tokens: int = 500
    temperature: float = 0.8
    top_k: int = 200
    repetition_penalty: float = 1.15
