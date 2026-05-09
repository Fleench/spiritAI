# SpiritAI

SpiritAI is a theological chat model built from scratch using a nanoGPT-based architecture.

## Architecture

The model is a causal transformer (decoder-only) with the following specifications:
- **Parameters:** ~124 Million
- **Layers:** 16
- **Embedding Dim:** 768
- **Attention Heads:** 12
- **Block Size:** 512
- **Vocab Size:** 50,257 (Standard GPT-2 BPE)

```text
Input ──► Embeddings ──► Transformer Blocks (x16) ──► LM Head ──► Output
                           │
                           ├──► LayerNorm
                           ├──► Scaled Dot Product Attention
                           ├──► LayerNorm
                           └──► MLP (GELU)
```

## Hardware Requirements
- **Inference:** Works on CPU or any GPU with >1GB VRAM.
- **Training:** Target hardware is an RTX PRO 4500 (32GB VRAM) or equivalent.
  - Set `GRAD_ACCUM_STEPS` in `.env` if you experience Out Of Memory (OOM) errors on smaller GPUs.

## Setup & Environment

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy the `.env.example` file and configure parameters:
   ```bash
   cp .env.example .env
   ```

### `.env` Reference
- `MAX_ITERS`: Total number of training steps (default: 500000)
- `GRAD_ACCUM_STEPS`: Number of micro-steps before an optimizer update (default: 4)

## Pipeline Walkthrough

The project uses a unified CLI for all operations.

1. **Download Data:**
   Fetches raw datasets from HuggingFace and static URLs.
   ```bash
   python cli.py download
   ```

   You can add arbitrary Hugging Face datasets in `spirit/data/huggingface_datasets.json` without changing code. The downloader auto-detects common Q&A/instruction schemas (for example `question`/`answer`, `prompt`/`response`, or `instruction`/`output`) and falls back to plain text columns when possible. Set `type` to `qa` or `text` when you want to override auto-detection, and use explicit column lists if a dataset has unusual names:
   ```json
   {
     "datasets": [
       {
         "path": "vericudebuget/Bible-responses-dataset-gotquestions",
         "split": "train",
         "type": "auto",
         "weight": 3,
         "output_file": "bible_responses_gotquestions.txt"
       },
       {
         "path": "owner/plain-text-corpus",
         "type": "text",
         "text_columns": ["body"],
         "max_rows": 50000
       },
       {
         "path": "owner/custom-qa-corpus",
         "type": "qa",
         "prompt_columns": ["question"],
         "response_columns": ["accepted_answer"]
       }
     ]
   }
   ```
   Custom downloads are written under `data/raw/huggingface/` and are included by `prepare`. To use a different config file for one run, pass `--hf-config path/to/file.json` to both `download` and `prepare`.
2. **Prepare Data:**
   Sanitizes text, applies weighting (e.g. Theological Q&A 3x and custom Hugging Face `weight` values), deduplicates, tokenizes, and saves to `.bin` format.
   ```bash
   python cli.py prepare
   ```
3. **Train:**
   Executes the training loop with linear warmup and cosine decay. Automatically resumes from checkpoint if available.
   ```bash
   python cli.py train
   ```
4. **Chat (Terminal):**
   ```bash
   python cli.py chat
   ```
5. **Web UI:**
   Launches a standard library `http.server` with a dark-mode chat interface and `/api/status` endpoint.
   ```bash
   python cli.py web --port 8000
   ```
6. **Check Status:**
   View dataset sizes, checkpoint metadata, and model parameter counts.
   ```bash
   python cli.py status
   ```

## Known Limitations & Future Ideas
- **Knowledge Boundaries:** Despite filtering, simple Wikipedia data may still occasionally bleed non-theological concepts.
- **Prompt Sensitivity:** The model has only been fine-tuned on instruction data matching a very specific formatting. Deviating from standard conversational structures may degrade response coherence.
- **Future Improvements:** Switch to full Megatron-LM/DeepSpeed for distributed training, implement RoPE (Rotary Position Embeddings), and utilize a custom theological token vocabulary.

## Legacy Archive
Original v1 code has been moved to `_archive/` for historical reference. See `_archive/README.md` for details.
