# SpiritAI Archive

This directory contains the original v1 implementation of SpiritAI, archived for reference before the total project rewrite.

## Files
- `data.py`: Legacy script to download Ante-Nicene Fathers and CPDV. Replaced by `spirit/data/sources.py`.
- `format_chat_corpus.py`: Legacy script to format conversational data. Replaced by `spirit/data/format.py`.
- `generate_chat_data.py`: Legacy script used to call the Gemini API for Q&A pairs.
- `model.py`: Legacy model implementation featuring manual attention and some hardcoded constraints. Replaced by `spirit/model.py`.
- `nano_gpt.py`: Legacy training loop script without gradient accumulation, proper warmup, or flash attention support. Replaced by `spirit/train/trainer.py`.
- `nano_gpt_chat.py`: Terminal chat interface. Replaced by `cli.py chat` subcommand.
- `nano_gpt_web.py`: Web server using standard library. Replaced by `web.py` with better status endpoints and UI.
- `paths.py`: Hardcoded path resolution. Replaced by unified configuration in `spirit/config.py`.
- `prepare_data.py`: Unstructured script for deduplication and tokenization. Replaced by `spirit/data/pipeline.py`.
- `sanatize.py`: Script to sanitize scraped data. Replaced by `spirit/data/pipeline.py`.
- `spirit_inference.py`: Generation logic that lacked proper stop-sequence handling. Replaced by `spirit/inference/generator.py`.
- `wiki.py`: Legacy simple wikipedia fetcher. Replaced by integrated pipeline fetching in `spirit/data/sources.py`.
