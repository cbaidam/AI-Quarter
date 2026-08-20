import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import streamlit as str_lib  # Přejmenováno, aby se nepletlo se zkratkou stoi

# =========================================================================
# ARCHITEKTURA MODELU CZECHIA AI QUARTER 1.2 (6.47M / 8 vrstev)
# =========================================================================
class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
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
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, None

    def generate(self, idx, max_new_tokens, temperature=0.6, top_k=40):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-4)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# =========================================================================
# NAČTENÍ MODELU S CACHOVÁNÍM (Aby se nenačítal při každém kliknutí znova)
# =========================================================================
# =========================================================================
# NAČTENÍ MODELU S CACHOVÁNÍM (Nekompromisní stahovač)
# =========================================================================
@str_lib.cache_resource
def load_my_model():
    checkpoint_file = "real_model.pt"
    
    # Pokud soubor neexistuje, nebo je to stará poškozená HTML stránka (pod 1MB), smažeme ho
    if os.path.exists(checkpoint_file) and os.path.getsize(checkpoint_file) < 1000000:
        try: os.remove(checkpoint_file)
        except: pass
            
    if not os.path.exists(checkpoint_file):
        import urllib.request
        url = (
            "https://"
            + "://github.com"
            + "cbaidam/"
            + "AI-Quarter/"
            + "releases/"
            + "download/"
            + "1.2-6.80Mil-B/"
            + "model.pt"
        )
        with str_lib.spinner("📥 Stahuji 43MB model Czechia AI Quarter 1.2 z cloudu..."):
            urllib.request.urlretrieve(url, checkpoint_file)
            
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    cfg = checkpoint["config"]
    stoi_dict = checkpoint["vocab"]["stoi"]
    itos_dict = {i: c for c, i in stoi_dict.items()}
    
    model_obj = GPTLanguageModel(
        vocab_size=cfg["vocab_size"],
        n_embd=cfg["n_embd"],
        block_size=cfg["block_size"],
        n_head=cfg["n_head"],
        n_layer=cfg["n_layer"]
    )
    model_obj.load_state_dict(checkpoint["model_state"])
    model_obj.eval()
    return model_obj, (stoi_dict, itos_dict)

# Spuštění stahování a načtení
model, vocabs = load_my_model()

# =========================================================================
# DESIGN WEBOVÉHO ROZHRANÍ (STREAMLIT CHAT)
# =========================================================================
str_lib.set_page_config(page_title="Czechia AI Quarter 1.2", page_icon="🤖")
str_lib.title("🤖 Czechia AI Quarter 1.2")
str_lib.caption("Vlastní Small Language Model (SLM) • 6.47M parametrů • 8 vrstev • 18M znaků z Wikipedie")

if model is None:
    str_lib.error("❌ Soubor 'model.pt' se nepodařilo z cloudu načíst!")
else:
    stoi, itos = vocabs
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: "".join(itos[i] for i in l)

    # Inicializace historie chatu v paměti webu
    if "messages" not in str_lib.session_state:
        str_lib.session_state.messages = []

    # Zobrazení předchozích zpráv z historie
    for message in str_lib.session_state.messages:
        with str_lib.chat_message(message["role"]):
            str_lib.markdown(message["content"])

    # Reakce na nový vstup od uživatele
    if podnet := str_lib.chat_input("Napište téma (např. Počítač, Člověk, AI)..."):
        # Zobrazení zprávy uživatele na webu
        with str_lib.chat_message("user"):
            str_lib.markdown(podnet)
        str_lib.session_state.messages.append({"role": "user", "content": podnet})

        # Generování odpovědi modelu
        with str_lib.chat_message("assistant"):
            encoded_input = encode(podnet)
            if not encoded_input:
                odpoved = "Omlouvám se, ale tato písmena můj slovník neobsahuje."
                str_lib.markdown(odpoved)
            else:
                context = torch.tensor([encoded_input], dtype=torch.long)
                with torch.no_grad():
                    generated = model.generate(context, max_new_tokens=400, temperature=0.6, top_k=40).tolist()
                
                generovany_zbytek = generated[len(encoded_input):]
                odpoved = decode(generovany_zbytek)
                str_lib.markdown(odpoved)
                
        str_lib.session_state.messages.append({"role": "assistant", "content": odpoved})

