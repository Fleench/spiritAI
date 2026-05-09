"""Chat with a trained SpiritAI Nano-GPT checkpoint in the terminal."""

from __future__ import annotations

from spirit_inference import SpiritGenerator


def main() -> None:
    generator = SpiritGenerator()
    info = generator.info
    settings = generator.settings

    print(f"Using device: {info.device}")
    print(
        f"Loaded SpiritAI: embd={info.n_embd}, heads={info.n_head}, layers={info.n_layer}, "
        f"block={info.block_size}, vocab={info.vocab_size}"
    )
    print(
        "Sampling: "
        f"top_k={settings.top_k}, temperature={settings.temperature}, "
        f"repetition_penalty={settings.repetition_penalty}"
    )
    print("Type 'exit' or 'quit' to stop.")
    print("-" * 60)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        print(f"AI: {generator.generate(user_input)}")


if __name__ == "__main__":
    main()
