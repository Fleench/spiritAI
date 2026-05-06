import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import re
import os
from dotenv import load_dotenv

# Load configuration from .env file
load_dotenv()

# Get settings from environment variables (must match training script)
OUTPUT_DIR = os.getenv('OUTPUT_DIR', '/workspace/models')
block_size = int(os.getenv('BLOCK_SIZE', '256'))
n_embd = int(os.getenv('N_EMBD', '256'))
n_head = int(os.getenv('N_HEAD', '8'))
n_layer = int(os.getenv('N_LAYER', '6'))
max_new_tokens = int(os.getenv('MAX_NEW_TOKENS', '100'))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")

# --- Load Vocabulary ---
print("Loading vocabulary...")
vocab_path = os.path.join(OUTPUT_DIR, 'vocab.json')
try:
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
except FileNotFoundError:
    print(f"Error: vocab.json not found at {vocab_path}")
    print(f"You must run nano_gpt.py first!")
    exit(1)

stoi = vocab_data['stoi']
# JSON saves integer keys as strings, so we must convert them back to integers
itos = {int(k): v for k, v in vocab_data['itos'].items()}
vocab_size = len(stoi)

encode = lambda s: [stoi[w] for w in re.findall(r"\w+|[^\w\s]", s.lower() if "david" not in s.lower() else s) if w in stoi]
decode = lambda l: " ".join([itos[i] for i in l])

# --- GPT Architecture (MUST MATCH TRAINING SCRIPT) ---
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

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
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) 
        return logits, None

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# --- Load Model Weights ---
print("Loading model weights...")
model = GPTLanguageModel()
model_path = os.path.join(OUTPUT_DIR, 'nano_gpt_model.pt')
try:
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
except FileNotFoundError:
    print(f"Error: nano_gpt_model.pt not found at {model_path}")
    print("You must run nano_gpt.py first!")
    exit(1)

m = model.to(device)
m.eval() # Set model to evaluation mode (turns off training mechanics)

print("\nModel Loaded! Ready to chat.")
print("Type 'exit' to stop.")
print("-" * 40)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
    if not user_input.strip():
        continue
    
    # Encode user input using our vocabulary
    context_idx = encode(user_input)
    if not context_idx:
        print("AI: [I didn't recognize any of those words.]")
        continue
        
    x = torch.tensor([context_idx], dtype=torch.long, device=device)
    
    # Generate tokens (amount from config)
    with torch.no_grad():
        y = m.generate(x, max_new_tokens=max_new_tokens)
    
    # The output includes our prompt, so we decode the whole thing
    out_text = decode(y[0].tolist())
    
    # Basic formatting to remove spaces before punctuation
    out_text = re.sub(r'\s+([.,;:!?])', r'\1', out_text)
    
    print(f"AI: {out_text}")
