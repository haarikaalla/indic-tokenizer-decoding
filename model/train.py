import torch
import torch.nn as nn
import sentencepiece as spm
import random
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.lm import TinyGPT

CORPUS_PATH = "data/corpus_hi.txt"
SP_MODEL = "tokenizer/hindi_bpe.model"
CKPT_PATH = "model/tinygpt_hindi.pt"
SEQ_LEN = 32
BATCH_SIZE = 64
EPOCHS = 15
LR = 3e-4

def load_data(sp):
    with open(CORPUS_PATH, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    random.seed(0)
    random.shuffle(lines)
    split = int(0.9 * len(lines))
    train_lines, val_lines = lines[:split], lines[split:]

    def encode_all(ls):
        ids = []
        for l in ls:
            ids.extend([sp.bos_id()] + sp.encode(l, out_type=int) + [sp.eos_id()])
        return torch.tensor(ids, dtype=torch.long)

    return encode_all(train_lines), encode_all(val_lines), val_lines

def get_batch(data, seq_len, batch_size):
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
    return x, y

def estimate_loss(model, data, seq_len, batch_size, n_batches=10):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_batch(data, seq_len, batch_size)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

def main():
    sp = spm.SentencePieceProcessor(model_file=SP_MODEL)
    vocab_size = sp.get_piece_size()
    print(f"Vocab size: {vocab_size}")

    train_data, val_data, _ = load_data(sp)
    print(f"Train tokens: {len(train_data)}, Val tokens: {len(val_data)}")

    model = TinyGPT(vocab_size=vocab_size, d_model=128, n_heads=4, n_layers=2, max_seq_len=SEQ_LEN)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    steps_per_epoch = len(train_data) // (SEQ_LEN * BATCH_SIZE) + 1
    for epoch in range(EPOCHS):
        for _ in range(steps_per_epoch):
            x, y = get_batch(train_data, SEQ_LEN, BATCH_SIZE)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = estimate_loss(model, train_data, SEQ_LEN, BATCH_SIZE)
        val_loss = estimate_loss(model, val_data, SEQ_LEN, min(BATCH_SIZE, len(val_data) - SEQ_LEN - 1))
        val_ppl = torch.exp(torch.tensor(val_loss)).item()
        print(f"epoch {epoch+1:2d}/{EPOCHS} | train_loss {train_loss:.3f} | "
              f"val_loss {val_loss:.3f} | val_perplexity {val_ppl:.2f}")

    torch.save({"model_state": model.state_dict(), "vocab_size": vocab_size,
                "d_model": 128, "n_heads": 4, "n_layers": 2, "max_seq_len": SEQ_LEN},
               CKPT_PATH)
    print(f"\nSaved checkpoint -> {CKPT_PATH}")

if __name__ == "__main__":
    main()
