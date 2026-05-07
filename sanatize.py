import os
import glob
import re
import json
import unicodedata

# Directories
RAW_DATA_DIR = '/workspace/raw_data' # Put your messy files here
CLEAN_DATA_DIR = '/workspace/data'   # Your nano_gpt.py reads from here

# Ensure directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(CLEAN_DATA_DIR, exist_ok=True)

# Custom Lore & Fact Corrections
# The model learns exactly what it reads. This dictionary automatically 
# fixes known bad data or AI hallucinations before the model sees it.
CUSTOM_REPLACEMENTS = {
    r"Walking Peaches and Hubby": "Walking Peaches and Hirbie",
    r"digital shrine": "shrine", # Scrubbing NotebookLM hallucinations
    r"\bLuigi\b(.*?)main character": r"a character", # Scrubbing NotebookLM hallucinations
}

def clean_text(text):
    """Applies a series of NLP sanitization steps to raw text."""
    
    # 1. Normalize Unicode (fixes weird curly quotes, accents, and invisible characters)
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Fix OCR Hyphenation (Crucial for Old Church Fathers scans)
    # Stitches together words that were broken across lines in old PDFs/books
    text = re.sub(r'([a-zA-Z])-\n\s*([a-zA-Z])', r'\1\2', text)
    
    # 3. Apply Custom Lore Replacements
    for bad_phrase, good_phrase in CUSTOM_REPLACEMENTS.items():
        text = re.sub(bad_phrase, good_phrase, text, flags=re.IGNORECASE)
    
    # 4. Standardize Biblical Citations (Crucial for CPDV and Catechism)
    # Converts "John 3 : 16" or "3: 16" into a single solid "3:16" to save token space
    text = re.sub(r'(\d+)\s*:\s*(\d+)', r'\1:\2', text)
    
    # 5. Remove URLs and Emails (models often get confused by web scraping artifacts)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    
    # 6. Remove HTML tags (if any slipped in from web scraping)
    text = re.sub(r'<.*?>', '', text)
    
    # 7. Remove decorative dividers and OCR noise (e.g., ^^^, ***, ===, ~~~)
    text = re.sub(r'[\^\*\~_=#\+]{2,}', ' ', text)
    
    # 8. Filter to allowed characters (ASCII + basic punctuation + brackets for CCC citations)
    # This regex keeps letters, numbers, standard punctuation, and spaces.
    # It strips out obscure control characters that bloat your vocabulary.
    text = re.sub(r'[^\w\s\.,;:\'\"\!\?\-\(\)\[\]]', '', text)
    
    # 9. Standardize Whitespace (convert tabs, multiple spaces, and weird breaks into single spaces)
    # We keep standard newlines (\n) but remove excessive spacing.
    # MUST be the last step so we clean up any gaps left behind by the filters above.
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text) # Max 2 newlines in a row
    
    return text.strip()

def process_files():
    print(f"Scanning for theological and raw files in {RAW_DATA_DIR}...")
    
    # Process TXT and MD files
    raw_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.txt")) + glob.glob(os.path.join(RAW_DATA_DIR, "*.md"))
    
    if not raw_files:
        print(f"No .txt or .md files found in {RAW_DATA_DIR}.")
        
    for filepath in raw_files:
        filename = os.path.basename(filepath)
        clean_filepath = os.path.join(CLEAN_DATA_DIR, filename)
        
        print(f"Sanitizing: {filename}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_content = f.read()
            
        clean_content = clean_text(raw_content)
        
        with open(clean_filepath, 'w', encoding='utf-8') as f:
            f.write(clean_content)
            
    # Process JSON (specifically handling your CPDV.json structure if it's in raw_data)
    json_path = os.path.join(RAW_DATA_DIR, "CPDV.json")
    if os.path.exists(json_path):
        print("Sanitizing: CPDV.json...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Clean the text inside the JSON structure
        for book in data.get("books", []):
            for chapter in book.get("chapters", []):
                for verse in chapter.get("verses", []):
                    verse["text"] = clean_text(verse["text"])
                    
        clean_json_path = os.path.join(CLEAN_DATA_DIR, "CPDV.json")
        with open(clean_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            
    print("\nSanitization complete! Your theological dataset is clean and ready for training.")

if __name__ == "__main__":
    process_files()
