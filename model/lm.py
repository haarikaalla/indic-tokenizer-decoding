"""
A small GPT-style (decoder-only Transformer) language model, built from scratch in PyTorch.
Intentionally tiny (2 layers, 128-dim) so it trains in seconds on CPU for this toy corpus --
the architecture pattern is identical to what scales up to real LLMs.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask (a token may only attend to the past)."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0:
            raise ValueError(f"d_model and n_heads must be positive, got {d_model} and {n_heads}.")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, n_heads, T, head_dim)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(causal_mask, float("-inf"))
        att = torch.softmax(att, dim=-1)

        out = att @ v  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: attention and feed-forward, each with a residual."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """Decoder-only Transformer LM: token+positional embeddings -> blocks -> logits."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}.")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}.")
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}.")
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Map token ids (B, T) to next-token logits (B, T, vocab_size)."""
        if idx.dim() != 2:
            raise ValueError(f"idx must be 2-D (batch, seq_len), got shape {tuple(idx.shape)}.")
        _, T = idx.shape
        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds the model context window of {self.max_seq_len}. "
                "Truncate the input (the decoding strategies do this automatically)."
            )
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)  # (B, T, vocab_size) raw logits
