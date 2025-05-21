import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# ---------- Hyperparameters ----------
batch_size = 64
block_size = 10  
max_iters = 4500
eval_interval = 500
learning_rate = 2e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 100
n_embd = 384
n_head = 4
n_layer = 4
dropout = 0.2

# ---------- Vocabulary ----------
chars = sorted(list(set("0123456789+= ")))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# ---------- Data Generation ----------

def generate_data(num_samples=45000):
    data = []
    for _ in range(num_samples):
        a = random.randint(0, 99)
        b = random.randint(0, 99)
        prompt = f"{a:02d}+{b:02d}="  
        answer = f"{a+b:04d}"          
        full_seq = prompt + answer    

        x = encode(full_seq)      
        y = x[1:] + [stoi[' ']]       
        data.append((x, y))
    return data

full_data = generate_data()
split = int(0.9 * len(full_data))
train_data = full_data[:split]
val_data = full_data[split:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data), (batch_size,))
    x = torch.tensor([data[i][0] for i in ix], dtype=torch.long)
    y = torch.tensor([data[i][1] for i in ix], dtype=torch.long)
    return x.to(device), y.to(device)

#------------Transformer Architechture---------------------

class SelfAttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
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

class AdditionGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is None:
            return logits, None

        # Compute masked loss ignoring padding spaces
        C = logits.size(-1)
        logits_flat = logits.view(B * T, C)
        targets_flat = targets.view(B * T)
        mask = (targets_flat != stoi[' '])
        if mask.sum() == 0:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        else:
            logits_masked = logits_flat[mask]
            targets_masked = targets_flat[mask]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.argmax(probs, dim=-1, keepdim=True)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


model = AdditionGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# -------------EVALUATION----------------

@torch.no_grad()
def eval_accuracy(model, num_samples=100):
    model.eval()
    correct = 0
    for _ in range(num_samples):
        a = random.randint(0, 99)
        b = random.randint(0, 99)
        prompt = f"{a:02d}+{b:02d}="
        true_answer = f"{a+b:04d}"
        input_ids = torch.tensor([encode(prompt)], device=device)
        out_ids = model.generate(input_ids, max_new_tokens=4)[0, -4:].tolist()
        pred_answer = decode(out_ids)
        if pred_answer == true_answer:
            correct += 1
    acc = correct / num_samples
    print(f"Eval accuracy on {num_samples} samples: {acc:.2%}")
    model.train()
    return acc

eval_accuracy(model, num_samples=100)

# ----------- TESTING THE MODEL ---------

# ========== TEST CASES ==========
test_cases = ["09+08=", "10+13=", "23+19=", "99+99=", "47+08=", "14+11=", "14+17=", "21+39=", "88+89="]
model.eval()
print("Before Saving:")
for prompt in test_cases:
    context = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    output = model.generate(context, max_new_tokens=4)
    pred = decode(output[0][-4:].tolist())
    print(f"{prompt}{pred}")

# ========== SAVE MODEL ==========
torch.save(model.state_dict(), "addition_gpt2.pt")

reloaded_model = AdditionGPT().to(device)
reloaded_model.load_state_dict(torch.load("addition_gpt2.pt"))
reloaded_model.eval()

print("\nAfter Loading:")
for prompt in test_cases:
    context = torch.tensor([encode(prompt)], dtype=torch.long).to(device)
    output = reloaded_model.generate(context, max_new_tokens=4)
    pred = decode(output[0][-4:].tolist())
    print(f"{prompt}{pred}")

