"""
Trains the binary sentiment classifier used as the generation-time safety filter.

Run from anywhere:  python classifier/train.py  (or python -m classifier.train)
"""
from __future__ import annotations

import logging
import random
import sys
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

if __package__ in (None, ""):  # `python classifier/train.py` -- put the repo root on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import loaders
from classifier.classifier_model import TextClassifier

logger = logging.getLogger(__name__)

Example = tuple[torch.Tensor, int]


def load_data(sp: spm.SentencePieceProcessor, data_path=None) -> tuple[list[Example], list[Example]]:
    """
    Parse the ``label<TAB>text`` dataset into tokenized (ids, label) pairs.

    Malformed or empty rows are skipped with a warning rather than aborting the run
    partway through -- a single bad line in a generated dataset should not cost a
    full training job.
    """
    path = config.require_file(
        data_path or config.SENTIMENT_DATA,
        "Generate the labeled data first: python classifier/generate_labeled_data.py",
    )
    examples: list[Example] = []
    skipped = 0
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split("\t", 1)
            if len(parts) != 2:
                skipped += 1
                logger.warning("%s:%d is not `label<TAB>text`; skipping.", path.name, lineno)
                continue
            label_raw, text = parts
            try:
                label = int(label_raw)
            except ValueError:
                skipped += 1
                logger.warning("%s:%d has non-integer label %r; skipping.", path.name, lineno, label_raw)
                continue
            if label not in (0, 1):
                skipped += 1
                logger.warning("%s:%d has out-of-range label %d; skipping.", path.name, lineno, label)
                continue
            ids = sp.encode(text, out_type=int)
            if not ids:
                skipped += 1
                continue
            examples.append((torch.tensor(ids, dtype=torch.long), label))

    if skipped:
        logger.warning("Skipped %d malformed rows in %s.", skipped, path)
    if len(examples) < 2:
        raise ValueError(f"{path} yielded {len(examples)} usable examples; need at least 2.")

    random.Random(config.SEED).shuffle(examples)
    split = max(1, int(0.85 * len(examples)))
    return examples[:split], examples[split:]


def collate(batch: list[Example]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a list of variable-length examples into one right-padded batch."""
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    return padded, lengths, torch.tensor(labels, dtype=torch.long)


def make_batches(data: list[Example], batch_size: int):
    """Yield shuffled, padded mini-batches. Shuffles a copy so the caller's list is untouched."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
    shuffled = list(data)
    random.shuffle(shuffled)
    for i in range(0, len(shuffled), batch_size):
        yield collate(shuffled[i:i + batch_size])


@torch.no_grad()
def evaluate(model: nn.Module, data: list[Example], batch_size: int = 32,
             device: torch.device | None = None) -> float:
    """Classification accuracy on `data`. Returns 0.0 for an empty split."""
    if not data:
        return 0.0
    was_training = model.training
    model.eval()
    correct, total = 0, 0
    for x, lengths, y in make_batches(data, batch_size):
        if device is not None:
            x, y = x.to(device), y.to(device)
        logits = model(x, lengths)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == y).sum().item()
        total += len(y)
    model.train(was_training)
    return correct / total if total else 0.0


def main() -> None:
    config.setup_logging()
    seed = config.seed_everything()
    device = config.resolve_device()
    logger.info("Training classifier on %s (seed=%d)", device, seed)

    sp = loaders.load_sentencepiece()
    train_data, val_data = load_data(sp)
    logger.info("Train: %d  Val: %d", len(train_data), len(val_data))

    model = TextClassifier(vocab_size=sp.get_piece_size()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.CLASSIFIER_LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.CLASSIFIER_EPOCHS):
        model.train()
        total_loss, n_batches = 0.0, 0
        for x, lengths, y in make_batches(train_data, config.CLASSIFIER_BATCH_SIZE):
            x, y = x.to(device), y.to(device)
            logits = model(x, lengths)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        val_acc = evaluate(model, val_data, config.CLASSIFIER_BATCH_SIZE, device=device)
        logger.info(
            "epoch %2d/%d | train_loss %.3f | val_acc %.1f%%",
            epoch + 1, config.CLASSIFIER_EPOCHS,
            total_loss / max(n_batches, 1), val_acc * 100,
        )

    saved = config.atomic_save(
        {k: v.cpu() for k, v in model.state_dict().items()},
        config.CLASSIFIER_CHECKPOINT,
    )
    logger.info("Saved classifier -> %s", saved)


if __name__ == "__main__":
    main()
