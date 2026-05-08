"""Chat with a trained SpiritAI Nano-GPT checkpoint."""

from __future__ import annotations

import os

import tiktoken

import torch

from model import GPTConfig, GPTLanguageModel
from paths import workspace_path

OUTPUT_DIR = workspace_path("models")
max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "160"))
temperature = float(os.getenv("TEMPERATURE", "0.8"))
top_k = int(os.getenv("TOP_K", "50"))
repetition_penalty = float(os.getenv("REPETITION_PENALTY", "1.15"))
device = "cuda" if torch.cuda.is_available() else "cpu"

config_path = OUTPUT_DIR / "config.json"
model_path = OUTPUT_DIR / "nano_gpt_model.pt"
GPT2_VOCAB_SIZE = 50_257
enc = tiktoken.get_encoding("gpt2")

if not config_path.exists():
    raise FileNotFoundError(f"config.json not found at {config_path}; run nano_gpt.py first")
if not model_path.exists():
    raise FileNotFoundError(f"nano_gpt_model.pt not found at {model_path}; run nano_gpt.py first")

config = GPTConfig.from_json(config_path)
config.vocab_size = GPT2_VOCAB_SIZE
model = GPTLanguageModel(config)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()


def encode(text: str) -> list[int]:
    return enc.encode(text)


def decode(ids: list[int]) -> str:
    return enc.decode([int(i) for i in ids])


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
