# Today, the model was born. Or yesterday. I don't know.
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size : int = 1024
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 12
    n_embd : int = 768
    dropout : float = 0.0
    bias : bool = True

#lets each token look backward at earlier tokens and decide what matters
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias = config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
    def forward(self, x):
        #B = Batch, T = sequence length, C = channel dimension (n_embd)
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim = 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim = -1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

#per-token feedforward network that processes each position after attention
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias = config.bias)
        self.gelu = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias = config.bias)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

#basically the glue with layer normalization and residual connections
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias = config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias = config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) #do an attention based update, but don't throw away old representation
        x = x + self.mlp(self.ln_2(x)) #normalize, process through MLP, add back to running representation
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        #wte = word token embeddings, which acts as a look up table that turns token IDs into vectors:  wpe = word positional embeddings, which encode where a token is in the sequence
        self.transformer = nn.ModuleDict(dict(
                                              wte = nn.Embedding(config.vocab_size, config.n_embd), 
                                              wpe = nn.Embedding(config.block_size, config.n_embd), 
                                              drop = nn.Dropout(config.dropout), 
                                              h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), 
                                              ln_f = nn.LayerNorm(config.n_embd, bias = config.bias)))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        #weight tying - lm_head and token embeddings share the same weight matrix
        self.transformer.wte.weight = self.lm_head.weight

        #intialize all custom weights with our custom scheme
        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean = 0.0, std = 0.02 / math.sqrt(2 * config.n_layer))
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)

    
    #idx is the input Token IDs
    def forward(self, idx, targets = None):
        device = idx.device
        B, T  = idx.size()
        #checks that sequence length is not longer than allowed context window
        assert T <= self.config.block_size, f"sequence_length {T} exceeds block_size {self.config.block_size}"
        pos = torch.arange(0, T, dtype = torch.long, device = device)
        tok_embd = self.transformer.wte(idx)
        pos_embd = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_embd + pos_embd)
        #each block gives the sequence another round of "look backward, mix information, then process more deeply"
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index = -1)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature = 1.0, top_k = None):
        for _ in range(max_new_tokens):
            #crop context to the last block_size tokens so we never exceed the model's window
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            #forward to get logits at every position, as we only care about the last one
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _  = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1)
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx