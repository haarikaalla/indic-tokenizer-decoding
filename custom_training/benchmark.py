"""
Benchmarks the from-scratch loss function and optimizer against PyTorch's built-ins,
training the SAME TinyGPT architecture on the SAME data, so the comparison is fair.

Two comparisons:
  A) torch.optim.AdamW           vs  SimpleAdamW (from scratch)
  B) F.cross_entropy             vs  LabelSmoothingLoss (from scratch)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

if __package__ in (None, ""):  # `python custom_training/benchmark.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import loaders
from custom_training.custom_optim import SimpleAdamW, LabelSmoothingLoss
from model.lm import TinyGPT
from model.train import get_batch, load_data

logger = logging.getLogger("benchmark")

SEQ_LEN = config.SEQ_LEN
BATCH_SIZE = config.BATCH_SIZE
EPOCHS = 10


def make_fresh_model(vocab_size: int, device: torch.device) -> TinyGPT:
    """Identical initialization across all runs, so the comparison is apples to apples."""
    torch.manual_seed(42)
    return TinyGPT(vocab_size=vocab_size, d_model=config.D_MODEL, n_heads=config.N_HEADS,
                   n_layers=config.N_LAYERS, max_seq_len=SEQ_LEN).to(device)


def train_run(model: TinyGPT, train_data: torch.Tensor, val_data: torch.Tensor,
              optimizer, loss_fn, label: str, device: torch.device) -> list[float]:
    """Train one configuration and return its per-epoch validation-loss trajectory."""
    steps_per_epoch = len(train_data) // (SEQ_LEN * BATCH_SIZE) + 1
    val_batch_size = max(1, min(BATCH_SIZE, len(val_data) - SEQ_LEN - 1))
    history: list[float] = []

    for _ in range(EPOCHS):
        model.train()
        for _ in range(steps_per_epoch):
            x, y = get_batch(train_data, SEQ_LEN, BATCH_SIZE, device=device)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vx, vy = get_batch(val_data, SEQ_LEN, val_batch_size, device=device)
            vlogits = model(vx)
            val_loss = F.cross_entropy(vlogits.reshape(-1, vlogits.size(-1)), vy.reshape(-1)).item()
        history.append(val_loss)

    logger.info("%-32s: val_loss trajectory = %s", label, [f"{h:.3f}" for h in history])
    return history


def main() -> None:
    config.setup_logging()
    device = config.resolve_device()

    sp = loaders.load_sentencepiece()
    vocab_size = sp.get_piece_size()
    train_data, val_data, _ = load_data(sp)

    logger.info("A) Optimizer comparison: torch.optim.AdamW vs. SimpleAdamW (from scratch)")
    m1 = make_fresh_model(vocab_size, device)
    train_run(m1, train_data, val_data, torch.optim.AdamW(m1.parameters(), lr=config.LEARNING_RATE),
              F.cross_entropy, "torch.optim.AdamW", device)

    m2 = make_fresh_model(vocab_size, device)
    train_run(m2, train_data, val_data, SimpleAdamW(list(m2.parameters()), lr=config.LEARNING_RATE),
              F.cross_entropy, "SimpleAdamW (from scratch)", device)

    logger.info("B) Loss comparison: F.cross_entropy vs. LabelSmoothingLoss (from scratch)")
    m3 = make_fresh_model(vocab_size, device)
    train_run(m3, train_data, val_data, torch.optim.AdamW(m3.parameters(), lr=config.LEARNING_RATE),
              F.cross_entropy, "F.cross_entropy", device)

    m4 = make_fresh_model(vocab_size, device)
    train_run(m4, train_data, val_data, torch.optim.AdamW(m4.parameters(), lr=config.LEARNING_RATE),
              LabelSmoothingLoss(smoothing=0.1), "LabelSmoothingLoss (from scratch)", device)

    logger.info(
        "Note: LabelSmoothingLoss values aren't directly comparable in scale to plain "
        "cross-entropy (different target distribution) -- the val_loss above is always "
        "measured with plain cross-entropy for a fair apples-to-apples read on both runs."
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
