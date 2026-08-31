"""
A text classifier built from scratch in PyTorch: token embeddings -> BiLSTM -> linear
head. Uses the same multilingual SentencePiece tokenizer as the generation pipeline,
so the whole system shares one vocabulary end to end.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    """Bidirectional-LSTM sequence classifier over subword ids."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        n_classes: int = 2,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}.")
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}.")
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) padded subword ids.
            lengths:   (B,) true (unpadded) length of each sequence, all >= 1.

        Returns:
            (B, n_classes) raw logits.
        """
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2-D (batch, seq_len), got {tuple(input_ids.shape)}.")
        if lengths.numel() != input_ids.size(0):
            raise ValueError(
                f"lengths has {lengths.numel()} entries but input_ids has batch size {input_ids.size(0)}."
            )
        if int(lengths.min()) < 1:
            # pack_padded_sequence raises an unhelpful error for zero-length rows
            raise ValueError("Every sequence must have length >= 1; got an empty sequence.")

        emb = self.embedding(input_ids)                                   # (B, T, E)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)                                    # h_n: (2, B, H)
        h_cat = torch.cat([h_n[0], h_n[1]], dim=-1)                        # (B, 2H): fwd+bwd
        return self.classifier(h_cat)                                      # (B, n_classes)
