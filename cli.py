"""Unified Command Line Interface for SpiritAI.

This script provides subcommands for downloading data, preparing datasets,
training the model, launching the CLI chat, and starting the web server.
"""

from __future__ import annotations

import argparse
import logging
import sys

from spirit.config import CHECKPOINT_PATH, TRAIN_BIN_PATH, VAL_BIN_PATH
from spirit.data.pipeline import prepare_dataset
from spirit.data.sources import fetch_all_sources
from spirit.train.trainer import train

# Configure logging for the application
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("spirit.cli")


def cmd_download(args: argparse.Namespace) -> None:
    """Download all required datasets."""
    failures = fetch_all_sources(args.hf_config)
    if failures:
        print("Could not download the following data sources:")
        for failure in failures:
            print(f"- {failure}")


def cmd_prepare(args: argparse.Namespace) -> None:
    """Sanitize, deduplicate, and tokenize the datasets."""
    prepare_dataset(args.hf_config)


def cmd_train(args: argparse.Namespace) -> None:
    """Train or resume training the NanoGPT model."""
    train()


def cmd_chat(args: argparse.Namespace) -> None:
    """Launch the terminal chat interface."""
    from spirit.inference.generator import SpiritGenerator

    try:
        generator = SpiritGenerator()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    print(f"Loaded SpiritAI on {generator.device}")
    print("Type 'exit' or 'quit' to stop.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            if not user_input:
                continue

            response = generator.generate(user_input)
            print(f"AI: {response}")
        except KeyboardInterrupt:
            break
        except (ValueError, RuntimeError) as e:
            logger.error(f"Error during generation: {e}")


def cmd_web(args: argparse.Namespace) -> None:
    """Launch the standalone web server UI."""
    import web
    web.start_server(args.port)


def cmd_status(args: argparse.Namespace) -> None:
    """Print the current project status and checkpoint information."""
    import os
    import torch

    print("=== SpiritAI Status ===")

    # Data status
    train_size = os.path.getsize(TRAIN_BIN_PATH) / (1024*1024) if TRAIN_BIN_PATH.exists() else 0
    val_size = os.path.getsize(VAL_BIN_PATH) / (1024*1024) if VAL_BIN_PATH.exists() else 0
    print("Data:")
    print(f"  Train Set: {train_size:.2f} MB")
    print(f"  Val Set:   {val_size:.2f} MB")

    # Checkpoint status
    if CHECKPOINT_PATH.exists():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        print("\nCheckpoint:")
        print(f"  Path: {CHECKPOINT_PATH}")
        print(f"  Iter: {ckpt.get('iter_num', 0)}")
        print(f"  Best Val Loss: {ckpt.get('best_val_loss', 'N/A')}")

        args_dict = ckpt.get('model_args', {})
        print("\nModel Architecture:")
        print(f"  Layers: {args_dict.get('n_layer')}")
        print(f"  Heads:  {args_dict.get('n_head')}")
        print(f"  Embed:  {args_dict.get('n_embd')}")
        print(f"  Block:  {args_dict.get('block_size')}")
        print(f"  Vocab:  {args_dict.get('vocab_size')}")
    else:
        print("\nCheckpoint: Not found. Run 'train' to start.")


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="SpiritAI Unified CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Download
    download_parser = subparsers.add_parser("download", help="Fetch all datasets")
    download_parser.add_argument(
        "--hf-config",
        default=None,
        help="Optional JSON file listing custom Hugging Face datasets",
    )

    # Prepare
    prepare_parser = subparsers.add_parser("prepare", help="Sanitize and tokenize datasets")
    prepare_parser.add_argument(
        "--hf-config",
        default=None,
        help="Optional JSON file listing custom Hugging Face datasets",
    )

    # Train
    subparsers.add_parser("train", help="Train or resume training")

    # Chat
    subparsers.add_parser("chat", help="Terminal chat interface")

    # Web
    web_parser = subparsers.add_parser("web", help="Launch web UI")
    web_parser.add_argument("--port", type=int, default=8000, help="Port to listen on")

    # Status
    subparsers.add_parser("status", help="Print checkpoint and data info")

    args = parser.parse_args()

    if args.command == "download":
        cmd_download(args)
    elif args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "web":
        cmd_web(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
