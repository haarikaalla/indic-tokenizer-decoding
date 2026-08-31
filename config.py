"""
Single source of truth for paths, hyperparameters, device selection and logging.

Every path is resolved relative to the repository root (not the current working
directory), so scripts behave identically whether they are launched from the repo
root, from a subdirectory, or from inside a container. Every value can be
overridden with an environment variable, which is what makes the same code usable
in a laptop run, a CI run and a container deployment without edits.

Environment overrides (all optional):
    ITD_LOG_LEVEL      INFO | DEBUG | WARNING ...
    ITD_DEVICE         auto | cpu | cuda | cuda:0
    ITD_TOKENIZER      path to the SentencePiece .model file
    ITD_LM_CKPT        path to the language-model checkpoint
    ITD_CLS_CKPT       path to the sentiment-classifier checkpoint
    ITD_SEED           global RNG seed
    ITD_CORS_ORIGINS   comma-separated allowed origins for the API (default: none)
"""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any

import torch

# --------------------------------------------------------------------------- #
# Repository layout
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parent

DATA_DIR: Path = PROJECT_ROOT / "data"
TOKENIZER_DIR: Path = PROJECT_ROOT / "tokenizer"
MODEL_DIR: Path = PROJECT_ROOT / "model"
CLASSIFIER_DIR: Path = PROJECT_ROOT / "classifier"
# NOTE: deliberately NOT named "compression" -- that shadows the Python 3.14 stdlib
# `compression` package, which breaks `import torch` (torch -> gzip -> compression._common).
QUANTIZATION_DIR: Path = PROJECT_ROOT / "quantization"


def _env_str(var: str, default: str) -> str:
    value = os.getenv(var)
    return value if value else default


def _env_path(var: str, default: Path) -> Path:
    """Resolve an env-var path against the repo root unless it is already absolute."""
    raw = os.getenv(var)
    if not raw:
        return default
    candidate = Path(raw).expanduser()
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)


def _env_int(var: str, default: int) -> int:
    raw = os.getenv(var)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logging.getLogger(__name__).warning("%s=%r is not an int; using %d", var, raw, default)
        return default


def _env_float(var: str, default: float) -> float:
    raw = os.getenv(var)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logging.getLogger(__name__).warning("%s=%r is not a float; using %s", var, raw, default)
        return default


# --------------------------------------------------------------------------- #
# Languages
# --------------------------------------------------------------------------- #
LANGUAGES: tuple[str, ...] = ("hi", "te", "ml", "kn")
LANG_TAGS: frozenset[str] = frozenset(f"<{code}>" for code in LANGUAGES)

# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
MULTILINGUAL_CORPUS: Path = _env_path("ITD_CORPUS", DATA_DIR / "corpus_multilingual.txt")
LANG_CORPUS: dict[str, Path] = {code: DATA_DIR / f"corpus_{code}.txt" for code in LANGUAGES}

TOKENIZER_PREFIX: Path = TOKENIZER_DIR / "multilingual_bpe"
TOKENIZER_MODEL: Path = _env_path("ITD_TOKENIZER", TOKENIZER_DIR / "multilingual_bpe.model")

LM_CHECKPOINT: Path = _env_path("ITD_LM_CKPT", MODEL_DIR / "tinygpt_multilingual.pt")
CLASSIFIER_CHECKPOINT: Path = _env_path("ITD_CLS_CKPT", CLASSIFIER_DIR / "sentiment_classifier.pt")
QUANTIZED_CHECKPOINT: Path = _env_path(
    "ITD_QUANTIZED_CKPT", QUANTIZATION_DIR / "tinygpt_multilingual_int8.pt"
)
SENTIMENT_DATA: Path = _env_path("ITD_SENTIMENT_DATA", CLASSIFIER_DIR / "sentiment_data.tsv")

# --------------------------------------------------------------------------- #
# Hyperparameters
# --------------------------------------------------------------------------- #
SEED: int = _env_int("ITD_SEED", 0)

TOKENIZER_VOCAB_SIZE: int = _env_int("ITD_TOKENIZER_VOCAB", 800)
PER_LANG_VOCAB_SIZE: int = _env_int("ITD_PER_LANG_VOCAB", 400)

SEQ_LEN: int = _env_int("ITD_SEQ_LEN", 32)
BATCH_SIZE: int = _env_int("ITD_BATCH_SIZE", 64)
EPOCHS: int = _env_int("ITD_EPOCHS", 20)
LEARNING_RATE: float = _env_float("ITD_LR", 3e-4)

D_MODEL: int = _env_int("ITD_D_MODEL", 160)
N_HEADS: int = _env_int("ITD_N_HEADS", 4)
N_LAYERS: int = _env_int("ITD_N_LAYERS", 3)

CLASSIFIER_EPOCHS: int = _env_int("ITD_CLS_EPOCHS", 10)
CLASSIFIER_BATCH_SIZE: int = _env_int("ITD_CLS_BATCH_SIZE", 32)
CLASSIFIER_LR: float = _env_float("ITD_CLS_LR", 1e-3)

# --------------------------------------------------------------------------- #
# Serving
# --------------------------------------------------------------------------- #
MAX_PROMPT_CHARS: int = _env_int("ITD_MAX_PROMPT_CHARS", 512)
MAX_NEW_TOKENS_LIMIT: int = _env_int("ITD_MAX_NEW_TOKENS", 100)


def cors_origins() -> list[str]:
    """Allowed CORS origins. Empty by default: no cross-origin access unless opted in."""
    raw = os.getenv("ITD_CORS_ORIGINS", "").strip()
    if not raw:
        return []
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


# --------------------------------------------------------------------------- #
# Runtime helpers
# --------------------------------------------------------------------------- #
_LOGGING_CONFIGURED = False


def setup_logging(level: str | int | None = None) -> None:
    """Configure root logging once. Safe to call from every entry point."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    resolved = level if level is not None else _env_str("ITD_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _LOGGING_CONFIGURED = True


def resolve_device(preferred: str | None = None) -> torch.device:
    """
    Pick a compute device. ``auto`` uses CUDA when available and falls back to CPU,
    so the exact same script runs on a laptop and on a GPU box.
    """
    name = (preferred or _env_str("ITD_DEVICE", "auto")).lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        logging.getLogger(__name__).warning("CUDA requested but unavailable; falling back to CPU.")
        name = "cpu"
    return torch.device(name)


def seed_everything(seed: int | None = None) -> int:
    """Seed Python and torch RNGs so runs are reproducible. Returns the seed used."""
    resolved = SEED if seed is None else seed
    random.seed(resolved)
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)
    return resolved


def require_file(path: Path | str, hint: str = "") -> Path:
    """
    Fail fast with an actionable message instead of a bare ``FileNotFoundError``
    from deep inside torch/sentencepiece.
    """
    resolved = Path(path)
    if not resolved.is_file():
        message = f"Required file not found: {resolved}"
        if hint:
            message = f"{message}\n  -> {hint}"
        raise FileNotFoundError(message)
    return resolved


def atomic_save(obj: Any, path: Path | str) -> Path:
    """
    Write a checkpoint to a temporary file then rename it into place, so an
    interrupted run can never leave a half-written checkpoint that later loads
    as a confusing corruption error.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target
