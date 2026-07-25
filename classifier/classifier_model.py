"""
A text classifier built from scratch in PyTorch: token embeddings -> BiLSTM -> linear
head. Uses the same multilingual SentencePiece tokenizer as the generation pipeline,
so the whole system shares one vocabulary end to end.
"""
import torch
import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, n_classes=2, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, input_ids, lengths):
        emb = self.embedding(input_ids)                                   # (B, T, E)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)                                    # h_n: (2, B, H)
        h_cat = torch.cat([h_n[0], h_n[1]], dim=-1)                        # (B, 2H): fwd+bwd
        return self.classifier(h_cat)                                      # (B, n_classes)
