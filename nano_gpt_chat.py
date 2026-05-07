"""Chat with a trained SpiritAI Nano-GPT checkpoint."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re

from dotenv import load_dotenv
import torch

from model import GPTConfig, GPTLanguageModel

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "/workspace/data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/workspace/models"))
max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "160"))
temperature = float(os.getenv("TEMPERATURE", "0.8"))
top_k = int(os.getenv("TOP_K", "50"))
repetition_penalty = float(os.getenv("REPETITION_PENALTY", "1.15"))
device = "cuda" if torch.cuda.is_available() else "cpu"

vocab_path = OUTPUT_DIR / "vocab.json"
if not vocab_path.exists():
    vocab_path = DATA_DIR / "vocab.json"
config_path = OUTPUT_DIR / "config.json"
model_path = OUTPUT_DIR / "nano_gpt_model.pt"

if not vocab_path.exists():
    raise FileNotFoundError(f"vocab.json not found at {vocab_path}; run prepare_data.py first")
if not config_path.exists():
    raise FileNotFoundError(f"config.json not found at {config_path}; run nano_gpt.py first")
if not model_path.exists():
    raise FileNotFoundError(f"nano_gpt_model.pt not found at {model_path}; run nano_gpt.py first")

with open(vocab_path, "r", encoding="utf-8") as f:
    vocab_data = json.load(f)
stoi = vocab_data["stoi"]
itos = {int(k): v for k, v in vocab_data["itos"].items()}

config = GPTConfig.from_json(config_path)
config.vocab_size = len(stoi)
model = GPTLanguageModel(config)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def encode(text: str) -> list[int]:
    unk = stoi.get("<unk>")
    ids: list[int] = []
    for token in TOKEN_RE.findall(text):
        if token in stoi:
            ids.append(stoi[token])
        elif token.lower() in stoi:
            ids.append(stoi[token.lower()])
        elif unk is not None:
            ids.append(unk)
    return ids


def decode(ids: list[int]) -> str:
    text = " ".join(itos[int(i)] for i in ids)
    text = re.sub(r"\s+([.,;:!?%\]\)])", r"\1", text)
    text = re.sub(r"([\[\(])\s+", r"\1", text)
    text = re.sub(r"\s+'\s+", "'", text)
    return text.strip()


print(f"Using device: {device}")
print(
    f"Loaded SpiritAI: embd={config.n_embd}, heads={config.n_head}, layers={config.n_layer}, "
    f"block={config.block_size}, vocab={config.vocab_size}"
)
print(f"Sampling: top_k={top_k}, temperature={temperature}, repetition_penalty={repetition_penalty}")
print("Type 'exit' or 'quit' to stop.")
print("-" * 60)

while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break
    if not user_input:
        continue

    prompt = f"{user_input}"
    context_idx = encode(prompt)
    if not context_idx:
        print("AI: [No recognizable tokens in prompt.]")
        continue

    x = torch.tensor([context_idx[-config.block_size :]], dtype=torch.long, device=device)
    with torch.no_grad():
        y = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
    print(f"AI: {decode(y[0].tolist())}")
