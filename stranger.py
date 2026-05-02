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