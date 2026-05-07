"""Train SpiritAI Nano-GPT on prepared theological token bins.

Fresh run target: RTX PRO 4500 / 32GB VRAM, 4-hour set-and-forget window.
Prepare data first:
    python prepare_data.py --input /workspace/raw_data/theology.txt --output-dir /workspace/data
Then train:
    python nano_gpt.py
"""

from __future__ import annotations

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import time

from dotenv import load_dotenv
import shutil
import torch

from model import GPTConfig, GPTLanguageModel

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "/workspace/data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/workspace/models"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# RTX PRO 4500 32GB target defaults.
batch_size = int(os.getenv("BATCH_SIZE", "64"))
block_size = int(os.getenv("BLOCK_SIZE", "256"))
max_iters = int(os.getenv("MAX_ITERS", "50000"))
eval_interval = int(os.getenv("EVAL_INTERVAL", "500"))
eval_iters = int(os.getenv("EVAL_ITERS", "100"))
learning_rate = float(os.getenv("LEARNING_RATE", "3e-4"))
min_lr = float(os.getenv("MIN_LR", "1e-5"))
weight_decay = float(os.getenv("WEIGHT_DECAY", "0.1"))
beta1 = float(os.getenv("BETA1", "0.9"))
beta2 = float(os.getenv("BETA2", "0.95"))
grad_clip = float(os.getenv("GRAD_CLIP", "1.0"))
compile_model = os.getenv("COMPILE", "true").lower() in {"1", "true", "yes"}
always_save_checkpoint = os.getenv("ALWAYS_SAVE_CHECKPOINT", "true").lower() in {"1", "true", "yes"}
seed = int(os.getenv("SEED", "1337"))

torch.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
if torch.cuda.is_available():
    torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if device == "cuda" else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
    os.getenv("DTYPE", "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16")
]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

train_bin = DATA_DIR / "train.bin"
val_bin = DATA_DIR / "val.bin"
vocab_path = DATA_DIR / "vocab.json"
if not train_bin.exists() or not val_bin.exists() or not vocab_path.exists():
    raise FileNotFoundError(
        f"Expected {train_bin}, {val_bin}, and {vocab_path}. Run prepare_data.py first."
    )

with open(vocab_path, "r", encoding="utf-8") as f:
    vocab_data = json.load(f)
vocab_size = len(vocab_data["stoi"])

train_data = torch.from_file(str(train_bin), shared=False, size=train_bin.stat().st_size // 4, dtype=torch.int32)
val_data = torch.from_file(str(val_bin), shared=False, size=val_bin.stat().st_size // 4, dtype=torch.int32)
if len(train_data) <= block_size or len(val_data) <= block_size:
    raise ValueError("train.bin/val.bin are too small for the configured block_size")

config = GPTConfig.from_env(vocab_size=vocab_size)
config.block_size = block_size
config.to_json(OUTPUT_DIR / "config.json")
shutil.copyfile(vocab_path, OUTPUT_DIR / "vocab.json")

print(f"Using device: {device} ({ptdtype})")
print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Tokens: train={len(train_data):,}, val={len(val_data):,}, vocab={vocab_size:,}")
print(
    "Config: "
    f"layers={config.n_layer}, heads={config.n_head}, embd={config.n_embd}, "
    f"block={config.block_size}, dropout={config.dropout}, batch={batch_size}"
)


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[int(i) : int(i) + block_size].long() for i in ix])
    y = torch.stack([data[int(i) + 1 : int(i) + 1 + block_size].long() for i in ix])
    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model: GPTLanguageModel) -> dict[str, float]:
    out: dict[str, float] = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            with ctx:
                _, loss = model(xb, yb)
            assert loss is not None
            losses[k] = loss.item()
        out[split] = float(losses.mean())
    model.train()
    return out


def get_lr(iter_num: int) -> float:
    if iter_num >= max_iters:
        return min_lr
    decay_ratio = iter_num / max_iters
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


model = GPTLanguageModel(config).to(device)
raw_model = model
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params / 1e6:.2f}M")
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
scaler = torch.amp.GradScaler(enabled=(device_type == "cuda" and ptdtype == torch.float16))

checkpoint_path = OUTPUT_DIR / "nano_gpt_checkpoint.pt"
model_path = OUTPUT_DIR / "nano_gpt_model.pt"
best_val_loss = float("inf")
start_iter = 0

if checkpoint_path.exists() and os.getenv("RESUME", "true").lower() in {"1", "true", "yes"}:
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_iter = int(checkpoint.get("iter", -1)) + 1
    best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
    print(f"Resuming from iteration {start_iter}; best val loss {best_val_loss:.4f}")

if compile_model and device_type == "cuda":
    print("Compiling model with torch.compile...")
    model = torch.compile(model)

print(f"Starting training for {max_iters:,} iterations")
t0 = time.time()
for iter_num in range(start_iter, max_iters):
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        losses = estimate_loss(raw_model)
        elapsed_hours = (time.time() - t0) / 3600
        print(
            f"step {iter_num:6d}: train {losses['train']:.4f}, val {losses['val']:.4f}, "
            f"lr {lr:.2e}, elapsed {elapsed_hours:.2f}h"
        )
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = min(best_val_loss, losses["val"])
            checkpoint = {
                "iter": iter_num,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "config": config.__dict__,
            }
            torch.save(checkpoint, checkpoint_path)
            torch.save(raw_model.state_dict(), model_path)
            config.to_json(OUTPUT_DIR / "config.json")
            print(f"saved checkpoint; best val loss {best_val_loss:.4f}")

    xb, yb = get_batch("train")
    with ctx:
        _, loss = model(xb, yb)
    assert loss is not None
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    if grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()

print("Training complete; saving final model")
torch.save(raw_model.state_dict(), model_path)
torch.save(
    {
        "iter": max_iters - 1,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": config.__dict__,
    },
    checkpoint_path,
)
config.to_json(OUTPUT_DIR / "config.json")
