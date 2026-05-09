"""Training loop for the SpiritAI model.

Handles gradient accumulation, learning rate schedules with warmup,
checkpointing, and data batching.
"""

from __future__ import annotations

import logging
import math
import os
import time

import numpy as np
import torch
from dotenv import load_dotenv

# Ensure dotenv is loaded before reading env vars
load_dotenv()

from spirit.config import CHECKPOINT_PATH, TRAIN_BIN_PATH, VAL_BIN_PATH, ModelConfig, TrainConfig
from spirit.model import GPT

logger = logging.getLogger(__name__)


def get_batch(data: np.ndarray, batch_size: int, block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of data.

    Args:
        data: Token data array.
        batch_size: Number of sequences in a batch.
        block_size: Length of each sequence.
        device: Target device string.

    Returns:
        A tuple of (x, y) tensors.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: np.ndarray,
    val_data: np.ndarray,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: str
) -> dict[str, float]:
    """Estimate the training and validation loss."""
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        if len(data) == 0:
            continue
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it: int, config: TrainConfig) -> float:
    """Compute learning rate with warmup and cosine decay."""
    # 1) linear warmup
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > config.lr_decay_iters:
        return config.min_lr
    # 3) in between, use cosine decay
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def train() -> None:
    """Execute the full training loop."""
    logger.info("Initializing training run...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load configs
    model_config = ModelConfig()
    train_config = TrainConfig()

    # Load data
    train_data = np.memmap(TRAIN_BIN_PATH, dtype=np.uint16, mode='r') if TRAIN_BIN_PATH.exists() else np.array([], dtype=np.uint16)
    val_data = np.memmap(VAL_BIN_PATH, dtype=np.uint16, mode='r') if VAL_BIN_PATH.exists() else np.array([], dtype=np.uint16)

    if len(train_data) == 0:
        logger.error("Training data not found. Run 'prepare' first.")
        return

    # Init or resume model
    iter_num = 0
    best_val_loss = 1e9

    model = GPT(model_config)

    if CHECKPOINT_PATH.exists():
        logger.info(f"Resuming from checkpoint {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

        # Validate vocab size
        ckpt_vocab_size = checkpoint['model_args']['vocab_size']
        if ckpt_vocab_size != model_config.vocab_size:
            logger.error(f"Vocabulary size mismatch! Checkpoint: {ckpt_vocab_size}, Config: {model_config.vocab_size}")
            return

        model.load_state_dict(checkpoint['model'])
        iter_num = checkpoint.get('iter_num', 0)
        best_val_loss = checkpoint.get('best_val_loss', 1e9)
    else:
        logger.info("Starting from scratch...")

    model.to(device)

    # Optimizer setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
        betas=(train_config.beta1, train_config.beta2)
    )

    if CHECKPOINT_PATH.exists() and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Compile model for faster training if supported
    if device == 'cuda' and hasattr(torch, "compile"):
        logger.info("Compiling model...")
        model = torch.compile(model) # type: ignore

    # Training loop
    X, Y = get_batch(train_data, train_config.batch_size, model_config.block_size, device)
    t0 = time.time()

    logger.info("Starting training...")
    while iter_num <= train_config.max_iters:
        # Determine current learning rate
        lr = get_lr(iter_num, train_config) if train_config.decay_lr else train_config.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Evaluation and checkpointing
        if iter_num % train_config.eval_interval == 0 and iter_num > 0:
            losses = estimate_loss(
                model, train_data, val_data, train_config.eval_iters,
                train_config.batch_size, model_config.block_size, device
            )
            logger.info(f"step {iter_num}: train loss {losses.get('train', 0.0):.4f}, val loss {losses.get('val', 0.0):.4f}")

            if losses.get('val', 0.0) < best_val_loss or train_config.always_save_checkpoint:
                best_val_loss = losses.get('val', best_val_loss)
                logger.info(f"Saving checkpoint to {CHECKPOINT_PATH}")
                checkpoint = {
                    'model': model.state_dict() if not hasattr(model, '_orig_mod') else model._orig_mod.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': {
                        'block_size': model_config.block_size,
                        'vocab_size': model_config.vocab_size,
                        'n_layer': model_config.n_layer,
                        'n_head': model_config.n_head,
                        'n_embd': model_config.n_embd,
                    },
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, CHECKPOINT_PATH)

        # Forward and backward pass with gradient accumulation
        for micro_step in range(train_config.grad_accum_steps):
            logits, loss = model(X, Y)
            # scale the loss to account for gradient accumulation
            loss = loss / train_config.grad_accum_steps

            # backward pass
            loss.backward()

            # fetch next batch
            X, Y = get_batch(train_data, train_config.batch_size, model_config.block_size, device)

        # Clip gradients
        if train_config.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % train_config.log_interval == 0:
            lossf = loss.item() * train_config.grad_accum_steps # scale back up for logging
            logger.info(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.4e}")

        iter_num += 1
