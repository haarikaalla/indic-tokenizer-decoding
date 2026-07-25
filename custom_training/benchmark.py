"""
Benchmarks the from-scratch loss function and optimizer against PyTorch's built-ins,
training the SAME TinyGPT architecture on the SAME data, so the comparison is fair.

Two comparisons:
  A) torch.optim.AdamW           vs  SimpleAdamW (from scratch)
  B) F.cross_entropy             vs  LabelSmoothingLoss (from scratch)
"""
import torch
import torch.nn.functional as F
import sentencepiece as spm
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.lm import TinyGPT
from model.train import load_data, get_batch
from custom_training.custom_optim import SimpleAdamW, LabelSmoothingLoss

SP_MODEL = "tokenizer/multilingual_bpe.model"
SEQ_LEN = 32
BATCH_SIZE = 64
EPOCHS = 10


def make_fresh_model(vocab_size):
    torch.manual_seed(42)  # identical init across all 4 runs for a fair comparison
    return TinyGPT(vocab_size=vocab_size, d_model=160, n_heads=4, n_layers=3, max_seq_len=SEQ_LEN)


def train_run(model, train_data, val_data, optimizer, loss_fn, label):
    steps_per_epoch = len(train_data) // (SEQ_LEN * BATCH_SIZE) + 1
    history = []
    for epoch in range(EPOCHS):
        for _ in range(steps_per_epoch):
            x, y = get_batch(train_data, SEQ_LEN, BATCH_SIZE)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vx, vy = get_batch(val_data, SEQ_LEN, min(BATCH_SIZE, len(val_data) - SEQ_LEN - 1))
            vlogits = model(vx)
            val_loss = F.cross_entropy(vlogits.reshape(-1, vlogits.size(-1)), vy.reshape(-1)).item()
        model.train()
        history.append(val_loss)
    print(f"{label:32s}: val_loss trajectory = {['%.3f' % h for h in history]}")
    return history


def main():
    sp = spm.SentencePieceProcessor(model_file=SP_MODEL)
    vocab_size = sp.get_piece_size()
    train_data, val_data, _ = load_data(sp)

    print("=" * 70)
    print("A) Optimizer comparison: torch.optim.AdamW vs. SimpleAdamW (from scratch)")
    print("=" * 70)
    m1 = make_fresh_model(vocab_size)
    opt1 = torch.optim.AdamW(m1.parameters(), lr=3e-4)
    train_run(m1, train_data, val_data, opt1, F.cross_entropy, "torch.optim.AdamW")

    m2 = make_fresh_model(vocab_size)
    opt2 = SimpleAdamW(list(m2.parameters()), lr=3e-4)
    train_run(m2, train_data, val_data, opt2, F.cross_entropy, "SimpleAdamW (from scratch)")

    print("\n" + "=" * 70)
    print("B) Loss comparison: F.cross_entropy vs. LabelSmoothingLoss (from scratch)")
    print("=" * 70)
    m3 = make_fresh_model(vocab_size)
    opt3 = torch.optim.AdamW(m3.parameters(), lr=3e-4)
    train_run(m3, train_data, val_data, opt3, F.cross_entropy, "F.cross_entropy")

    m4 = make_fresh_model(vocab_size)
    opt4 = torch.optim.AdamW(m4.parameters(), lr=3e-4)
    ls_loss = LabelSmoothingLoss(smoothing=0.1)
    train_run(m4, train_data, val_data, opt4, ls_loss, "LabelSmoothingLoss (from scratch)")
    print("\nNote: LabelSmoothingLoss values aren't directly comparable in scale to plain "
          "cross-entropy (different target distribution) -- final val_loss above is always "
          "measured with plain cross-entropy for a fair apples-to-apples read on both runs.")


if __name__ == "__main__":
    main()
