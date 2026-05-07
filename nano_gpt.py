import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import json
import re
import os
import glob
from dotenv import load_dotenv

# Set memory management flags for PyTorch to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Clear CUDA cache immediately to free up memory from previous crashes
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Load configuration from .env file
load_dotenv()

# Get settings from environment variables (with defaults if not set)
DATA_DIR = os.getenv('DATA_DIR', '/workspace/data')
OUTPUT_DIR = os.getenv('OUTPUT_DIR', '/workspace/models')
batch_size = int(os.getenv('BATCH_SIZE', '16'))
block_size = int(os.getenv('BLOCK_SIZE', '256'))
max_iters = int(os.getenv('MAX_ITERS', '10000'))
eval_interval = int(os.getenv('EVAL_INTERVAL', '500'))
learning_rate = float(os.getenv('LEARNING_RATE', '3e-4'))

# Adjusting architecture slightly for safety: 512 is still high quality
n_embd = int(os.getenv('N_EMBD', '512')) 
n_head = int(os.getenv('N_HEAD', '8'))
n_layer = int(os.getenv('N_LAYER', '12'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Using device: {device}")
print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

print("\n1. Loading and Tokenizing Dataset...")
raw_text = []

# Load original JSON if it exists
bible_path = os.path.join(DATA_DIR, "CPDV.json")
if os.path.exists(bible_path):
    print(f" - Loading {bible_path}")
    with open(bible_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for book in data.get("books", []):
        for chapter in book.get("chapters", []):
            for verse in chapter.get("verses", []):
                raw_text.append(verse["text"])

# Load all .txt and .md files
for filepath in glob.glob(os.path.join(DATA_DIR, "*.txt")) + glob.glob(os.path.join(DATA_DIR, "*.md")):
    print(f" - Loading {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text.append(f.read())

if not raw_text:
    print(f"Error: No text data found in {DATA_DIR}")
    exit(1)

full_text = " ".join(raw_text)

# Basic word/punctuation tokenizer
tokens = re.findall(r"\w+|[^\w\s]", full_text)
vocab = sorted(list(set(tokens)))
vocab_size = len(vocab)
print(f"Total tokens in dataset: {len(tokens)}")
print(f"Vocabulary size: {vocab_size}")

# Create mappings from word to integer and integer to word
stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for i, w in enumerate(vocab)}
encode = lambda s: [stoi[w] for w in re.findall(r"\w+|[^\w\s]", s) if w in stoi]
decode = lambda l: " ".join([itos[i] for i in l])

# Train/Test Split
print("2. Converting to Tensors...")
data_tensor = torch.tensor([stoi[w] for w in tokens], dtype=torch.long)
n = int(0.9 * len(data_tensor))
train_data = data_tensor[:n]
val_data = data_tensor[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(50)
        for k in range(50):
            X, Y = get_batch(split)
            with torch.amp.autocast('cuda'):
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- GPT Architecture ---

class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Skip connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb
        
        # Applying Gradient Checkpointing to save massive VRAM
        for block in self.blocks:
            if self.training:
                # Use dummy inputs to ensure it works correctly with checkpoint
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 0.01)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- Training ---
print("\n3. Initializing SpiritAI Nano-GPT...")
model = GPTLanguageModel()
m = model.to(device)

# Print number of parameters
n_params = sum(p.numel() for p in m.parameters())
print(f"Model Parameters: {n_params / 1e6:.2f} Million")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scaler = torch.amp.GradScaler('cuda') 

print(f"\n4. Starting Training for {max_iters} iterations on {device.upper()}...")
for iter in range(max_iters):
    # Aggressive memory cleanup
    if iter % 100 == 0:
        torch.cuda.empty_cache()

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model)
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    
    with torch.amp.autocast('cuda'):
        logits, loss = model(xb, yb)
        
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

print("\n--- Training Complete! Saving Model ---")
model_path = os.path.join(OUTPUT_DIR, 'nano_gpt_model.pt')
torch.save(model.state_dict(), model_path)
print(f"Model saved to '{model_path}'")

vocab_path = os.path.join(OUTPUT_DIR, 'vocab.json')
vocab_data = {'stoi': stoi, 'itos': itos}
with open(vocab_path, 'w', encoding='utf-8') as f:
    json.dump(vocab_data, f)
print(f"Vocabulary saved to '{vocab_path}'")
