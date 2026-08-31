"""
Safe, shared loaders for the trained artifacts (tokenizer, language model,
classifier).

Before this module existed, the same "torch.load -> rebuild TinyGPT -> load_state_dict"
block was copy-pasted into five scripts, each with slightly different (or missing)
error handling. Centralising it means:

* one place enforces ``weights_only=True`` -- checkpoints are deserialised as pure
  tensor data, never as arbitrary pickled Python objects (untrusted checkpoints are
  a real remote-code-execution vector);
* one place validates the checkpoint schema, so a truncated or mismatched file
  produces an actionable error instead of a ``KeyError: 'd_model'``;
* one place handles device placement, so CPU/GPU behaviour is consistent everywhere.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

import config
from classifier.classifier_model import TextClassifier
from model.lm import TinyGPT

if TYPE_CHECKING:  # pragma: no cover - typing only
    import sentencepiece as spm

logger = logging.getLogger(__name__)

REQUIRED_LM_KEYS: tuple[str, ...] = (
    "model_state", "vocab_size", "d_model", "n_heads", "n_layers", "max_seq_len",
)

TRAIN_FIRST_HINT = (
    "Run the training pipeline first: python -m data.generate_corpus && "
    "python -m tokenizer.train_sentencepiece && python -m model.train && "
    "python -m classifier.generate_labeled_data && python -m classifier.train"
)


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    """Deserialise a checkpoint as weights-only data, with a readable failure mode."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:  # corrupted file, truncated write, torch version skew
        raise RuntimeError(f"Failed to load checkpoint {path}: {exc}") from exc


def load_sentencepiece(model_path: Path | str | None = None) -> "spm.SentencePieceProcessor":
    """Load the SentencePiece tokenizer, verifying the model file exists first.

    ``sentencepiece`` is imported lazily so that checkpoint-only code paths (and the
    tests covering them) still work in environments where the wheel is unavailable.
    """
    path = config.require_file(model_path or config.TOKENIZER_MODEL, TRAIN_FIRST_HINT)
    try:
        import sentencepiece as spm
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "sentencepiece is not installed. Install it with: pip install sentencepiece"
        ) from exc
    try:
        return spm.SentencePieceProcessor(model_file=str(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to load SentencePiece model {path}: {exc}") from exc


def load_language_model(
    ckpt_path: Path | str | None = None,
    device: torch.device | str | None = None,
) -> TinyGPT:
    """
    Rebuild ``TinyGPT`` from a checkpoint and put it in eval mode on ``device``.

    The architecture hyperparameters are read from the checkpoint itself, so a model
    trained with different dimensions still loads correctly.
    """
    path = config.require_file(ckpt_path or config.LM_CHECKPOINT, TRAIN_FIRST_HINT)
    target_device = torch.device(device) if device is not None else config.resolve_device()

    ckpt = _load_checkpoint(path, target_device)
    missing = [key for key in REQUIRED_LM_KEYS if key not in ckpt]
    if missing:
        raise RuntimeError(
            f"Checkpoint {path} is missing required keys {missing}. "
            "It was probably written by an incompatible version of model/train.py."
        )

    model = TinyGPT(
        vocab_size=ckpt["vocab_size"],
        d_model=ckpt["d_model"],
        n_heads=ckpt["n_heads"],
        n_layers=ckpt["n_layers"],
        max_seq_len=ckpt["max_seq_len"],
    )
    try:
        model.load_state_dict(ckpt["model_state"])
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint {path} does not match the current TinyGPT architecture: {exc}"
        ) from exc

    model.to(target_device).eval()
    logger.debug("Loaded language model from %s onto %s", path, target_device)
    return model


def load_classifier(
    vocab_size: int,
    ckpt_path: Path | str | None = None,
    device: torch.device | str | None = None,
) -> TextClassifier:
    """Rebuild the sentiment ``TextClassifier`` and put it in eval mode on ``device``."""
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}.")

    path = config.require_file(ckpt_path or config.CLASSIFIER_CHECKPOINT, TRAIN_FIRST_HINT)
    target_device = torch.device(device) if device is not None else config.resolve_device()

    state = _load_checkpoint(path, target_device)
    model = TextClassifier(vocab_size=vocab_size)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Classifier checkpoint {path} does not match the current TextClassifier "
            f"(vocab_size={vocab_size}): {exc}"
        ) from exc

    model.to(target_device).eval()
    logger.debug("Loaded classifier from %s onto %s", path, target_device)
    return model
