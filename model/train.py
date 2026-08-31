"""
Trains the multilingual TinyGPT language model on the language-tagged corpus.

Run from anywhere:  python model/train.py
Everything (paths, hyperparameters, device, seed) comes from config.py and can be
overridden with environment variables -- no code edits needed to retune a run.
"""
from __future__ import annotations

import logging
import math
import random
import sys
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn

if __package__ in (None, ""):  # `python model/train.py` -- put the repo root on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import loaders
from model.lm import TinyGPT

logger = logging.getLogger(__name__)


def load_data(
    sp: spm.SentencePieceProcessor,
    corpus_path=None,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """
    Tokenize the corpus into one flat id stream per split.

    Returns:
        (train_ids, val_ids, val_lines) -- a 90/10 split, shuffled with a fixed seed
        so training and evaluation always see the same partition.
    """
    path = config.require_file(
        corpus_path or config.MULTILINGUAL_CORPUS,
        "Generate the corpora first: python data/generate_corpus.py",
    )
    with path.open(encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"Corpus {path} is empty; nothing to train on.")

    random.Random(config.SEED).shuffle(lines)
    split = int(0.9 * len(lines))
    if split == 0 or split == len(lines):
        raise ValueError(
            f"Corpus {path} has only {len(lines)} lines -- too few to make a train/val split."
        )
    train_lines, val_lines = lines[:split], lines[split:]

    def encode_all(subset: list[str]) -> torch.Tensor:
        ids: list[int] = []
        for line in subset:
            ids.extend([sp.bos_id()] + sp.encode(line, out_type=int) + [sp.eos_id()])
        return torch.tensor(ids, dtype=torch.long)

    return encode_all(train_lines), encode_all(val_lines), val_lines


def get_batch(
    data: torch.Tensor,
    seq_len: int,
    batch_size: int,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `batch_size` random (input, next-token-target) windows of length `seq_len`."""
    max_start = len(data) - seq_len - 1
    if max_start < 1:
        raise ValueError(
            f"Split has {len(data)} tokens but seq_len={seq_len} needs at least {seq_len + 2}. "
            "Use a shorter ITD_SEQ_LEN or a larger corpus."
        )
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
    if device is not None:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    data: torch.Tensor,
    seq_len: int,
    batch_size: int,
    n_batches: int = 10,
    device: torch.device | None = None,
) -> float:
    """Average cross-entropy over `n_batches` random windows, with the model in eval mode."""
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, seq_len, batch_size, device=device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(loss.item())
    model.train(was_training)
    return sum(losses) / len(losses)


def main() -> None:
    config.setup_logging()
    seed = config.seed_everything()
    device = config.resolve_device()
    logger.info("Training on %s (seed=%d)", device, seed)

    sp = loaders.load_sentencepiece()
    vocab_size = sp.get_piece_size()
    logger.info("Vocab size: %d", vocab_size)

    train_data, val_data, _ = load_data(sp)
    logger.info("Train tokens: %d, Val tokens: %d", len(train_data), len(val_data))

    seq_len, batch_size = config.SEQ_LEN, config.BATCH_SIZE
    val_batch_size = max(1, min(batch_size, len(val_data) - seq_len - 1))

    model = TinyGPT(vocab_size=vocab_size, d_model=config.D_MODEL, n_heads=config.N_HEADS,
                    n_layers=config.N_LAYERS, max_seq_len=seq_len).to(device)
    logger.info("Model params: %s", f"{sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    steps_per_epoch = len(train_data) // (seq_len * batch_size) + 1
    for epoch in range(config.EPOCHS):
        model.train()
        for _ in range(steps_per_epoch):
            x, y = get_batch(train_data, seq_len, batch_size, device=device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        train_loss = estimate_loss(model, train_data, seq_len, batch_size, device=device)
        val_loss = estimate_loss(model, val_data, seq_len, val_batch_size, device=device)
        val_ppl = math.exp(min(val_loss, 700))  # guard against overflow on a diverged run
        logger.info(
            "epoch %2d/%d | train_loss %.3f | val_loss %.3f | val_perplexity %.2f",
            epoch + 1, config.EPOCHS, train_loss, val_loss, val_ppl,
        )

    saved = config.atomic_save(
        {"model_state": {k: v.cpu() for k, v in model.state_dict().items()},
         "vocab_size": vocab_size,
         "d_model": config.D_MODEL, "n_heads": config.N_HEADS,
         "n_layers": config.N_LAYERS, "max_seq_len": seq_len},
        config.LM_CHECKPOINT,
    )
    logger.info("Saved checkpoint -> %s", saved)


if __name__ == "__main__":
    main()
