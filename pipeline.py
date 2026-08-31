"""
A clean, object-oriented facade over the whole pipeline (tokenizer -> LM -> decoding
-> safety filter). The individual scripts (tokenizer/, model/, decoding/, classifier/)
remain as standalone, runnable steps -- this module composes them into a single
reusable interface, the way a real production pipeline would be organized.

Demonstrates: clear separation of concerns, a consistent interface across stages,
and dependency injection (each component receives what it needs, no global state).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import sentencepiece as spm
import torch

import config
from classifier.classifier_model import TextClassifier
from decoding.strategies import (
    greedy_decode, beam_search_decode, top_k_sampling_decode, top_p_sampling_decode
)
from model.lm import TinyGPT

logger = logging.getLogger(__name__)


class DecodingStrategy(ABC):
    """Common interface so any strategy can be swapped in without changing caller code."""
    @abstractmethod
    def decode(self, model: TinyGPT, input_ids: torch.Tensor, max_new_tokens: int, eos_id: int) -> torch.Tensor:
        ...


class Greedy(DecodingStrategy):
    def decode(self, model: TinyGPT, input_ids: torch.Tensor, max_new_tokens: int, eos_id: int) -> torch.Tensor:
        return greedy_decode(model, input_ids, max_new_tokens, eos_id=eos_id)


class BeamSearch(DecodingStrategy):
    def __init__(self, beam_width: int = 4):
        if beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {beam_width}.")
        self.beam_width = beam_width

    def decode(self, model: TinyGPT, input_ids: torch.Tensor, max_new_tokens: int, eos_id: int) -> torch.Tensor:
        return beam_search_decode(model, input_ids, max_new_tokens, beam_width=self.beam_width, eos_id=eos_id)


class TopK(DecodingStrategy):
    def __init__(self, k: int = 10, temperature: float = 0.8):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")
        self.k, self.temperature = k, temperature

    def decode(self, model: TinyGPT, input_ids: torch.Tensor, max_new_tokens: int, eos_id: int) -> torch.Tensor:
        return top_k_sampling_decode(model, input_ids, max_new_tokens, k=self.k,
                                      temperature=self.temperature, eos_id=eos_id)


class TopP(DecodingStrategy):
    def __init__(self, p: float = 0.9, temperature: float = 0.8):
        if not 0.0 < p <= 1.0:
            raise ValueError(f"p must be in (0, 1], got {p}.")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")
        self.p, self.temperature = p, temperature

    def decode(self, model: TinyGPT, input_ids: torch.Tensor, max_new_tokens: int, eos_id: int) -> torch.Tensor:
        return top_p_sampling_decode(model, input_ids, max_new_tokens, p=self.p,
                                      temperature=self.temperature, eos_id=eos_id)


STRATEGY_NAMES: tuple[str, ...] = ("greedy", "beam", "top_k", "top_p")


def build_strategy(name: str) -> DecodingStrategy:
    """Instantiate a decoding strategy by name, with the project's default settings."""
    defaults: dict[str, DecodingStrategy] = {
        "greedy": Greedy(),
        "beam": BeamSearch(beam_width=4),
        "top_k": TopK(k=10, temperature=0.8),
        "top_p": TopP(p=0.9, temperature=0.8),
    }
    if name not in defaults:
        raise ValueError(f"Unknown strategy {name!r}. Choose one of {sorted(defaults)}.")
    return defaults[name]


class IndicTokenizer:
    """Thin wrapper around SentencePiece so callers depend on our interface, not the library directly."""
    def __init__(self, model_path: str | Path):
        path = config.require_file(model_path, "Train the tokenizer: python tokenizer/train_sentencepiece.py")
        self.model_path = path
        self.sp = spm.SentencePieceProcessor(model_file=str(path))

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    @property
    def bos_id(self) -> int:
        return self.sp.bos_id()

    @property
    def eos_id(self) -> int:
        return self.sp.eos_id()


class SafetyFilter:
    """Wraps the sentiment classifier as a pluggable generation-time policy check."""
    def __init__(self, classifier: TextClassifier, tokenizer: IndicTokenizer, accept_label: int = 1):
        self.classifier = classifier
        self.tokenizer = tokenizer
        self.accept_label = accept_label

    def passes(self, text: str) -> bool:
        """True when the candidate text satisfies the policy. Unscoreable text is rejected."""
        encoded = self.tokenizer.encode(text)
        if not encoded:
            # An empty encoding cannot be scored; fail closed rather than waving it through.
            return False
        device = next(self.classifier.parameters()).device
        ids = torch.tensor([encoded], dtype=torch.long, device=device)
        lengths = torch.tensor([ids.shape[1]])
        with torch.no_grad():
            logits = self.classifier(ids, lengths)
        return int(torch.argmax(logits, dim=-1).item()) == self.accept_label


class MultilingualGenerationPipeline:
    """
    The single entry point a caller interacts with. Composes tokenizer + model +
    decoding strategy + optional safety filter into one `.generate()` call.
    """
    def __init__(self, model: TinyGPT, tokenizer: IndicTokenizer,
                 safety_filter: SafetyFilter | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.safety_filter = safety_filter

    @property
    def device(self) -> torch.device:
        """Device the model lives on; inputs are always built here to avoid a device mismatch."""
        return next(self.model.parameters()).device

    def generate(self, prompt: str, strategy: DecodingStrategy, max_new_tokens: int = 20,
                 max_filter_attempts: int = 5, seed: int | None = None) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}.")
        if max_filter_attempts < 1:
            raise ValueError(f"max_filter_attempts must be >= 1, got {max_filter_attempts}.")

        prompt_ids = [self.tokenizer.bos_id] + self.tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        text = ""
        for attempt in range(max_filter_attempts):
            if seed is not None:
                torch.manual_seed(seed + attempt)
            out_ids = strategy.decode(self.model, input_ids, max_new_tokens, self.tokenizer.eos_id)
            text = self.tokenizer.decode(out_ids[0].tolist())

            if self.safety_filter is None or self.safety_filter.passes(text):
                return text

        logger.warning(
            "Safety filter rejected all %d candidates for prompt %r; returning the last one.",
            max_filter_attempts, prompt,
        )
        return text  # exhausted attempts; return the last candidate anyway


def build_pipeline(
    with_safety_filter: bool = True,
    device: torch.device | str | None = None,
) -> MultilingualGenerationPipeline:
    """
    Assemble the default pipeline from the configured checkpoints.

    This is the one place that knows how the production artifacts fit together, so
    scripts and the API no longer each re-implement loading (and each get it
    slightly differently wrong).
    """
    import loaders  # imported here to keep module import cheap and avoid a cycle

    target_device = torch.device(device) if device is not None else config.resolve_device()
    tokenizer = IndicTokenizer(config.TOKENIZER_MODEL)
    model = loaders.load_language_model(device=target_device)

    safety_filter = None
    if with_safety_filter:
        classifier = loaders.load_classifier(tokenizer.vocab_size, device=target_device)
        safety_filter = SafetyFilter(classifier, tokenizer)

    return MultilingualGenerationPipeline(model, tokenizer, safety_filter)


if __name__ == "__main__":
    config.setup_logging()
    pipeline = build_pipeline()

    logger.info("OOP pipeline demo -- same interface, swappable decoding strategy:")
    for name in ("greedy", "beam", "top_k", "top_p"):
        generated = pipeline.generate("<te> విద్యార్థి", build_strategy(name), max_new_tokens=15, seed=0)
        logger.info("  %-12s: %s", name, generated)
