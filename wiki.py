import pandas as pd
import os

# Hardcoded source and destination
URL = "https://huggingface.co/datasets/rahular/simple-wikipedia/resolve/main/data/train-00000-of-00001-090b52ccb189d47a.parquet"
# Use the absolute path to your raw_data directory
OUTPUT_PATH = "/workspace/fleench/spiritai/raw_data/wikipedia_simple.txt"

def extract():
    print(f"Downloading from HuggingFace...")
    # Read the parquet directly into memory
    df = pd.read_parquet(URL)
    
    print(f"Extracting text column...")
    # Join the 'text' column into one massive string
    raw_text = "\n\n".join(df['text'].astype(str).tolist())
    
    # Ensure directory exists just in case
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(raw_text)
    
    print(f"Success! Saved {len(raw_text):,} characters to {OUTPUT_PATH}")

if __name__ == "__main__":
    extract()
