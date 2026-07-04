"""
A clean, object-oriented facade over the whole pipeline (tokenizer -> LM -> decoding
-> safety filter). The individual scripts (tokenizer/, model/, decoding/, classifier/)
remain as standalone, runnable steps -- this module composes them into a single
reusable interface, the way a real production pipeline would be organized.

Demonstrates: clear separation of concerns, a consistent interface across stages,
and dependency injection (each component receives what it needs, no global state).
"""
from __future__ import annotations
import torch
import sentencepiece as spm
from abc import ABC, abstractmethod

from model.lm import TinyGPT
from classifier.classifier_model import TextClassifier
from decoding.strategies import (
    greedy_decode, beam_search_decode, top_k_sampling_decode, top_p_sampling_decode
)


class DecodingStrategy(ABC):
    """Common interface so any strategy can be swapped in without changing caller code."""
    @abstractmethod
    def decode(self, model: TinyGPT, input_ids: torch.Tensor, max_new_tokens: int, eos_id: int) -> torch.Tensor:
        ...


class Greedy(DecodingStrategy):
    def decode(self, model, input_ids, max_new_tokens, eos_id):
        return greedy_decode(model, input_ids, max_new_tokens, eos_id=eos_id)


class BeamSearch(DecodingStrategy):
    def __init__(self, beam_width: int = 4):
        self.beam_width = beam_width

    def decode(self, model, input_ids, max_new_tokens, eos_id):
        return beam_search_decode(model, input_ids, max_new_tokens, beam_width=self.beam_width, eos_id=eos_id)


class TopK(DecodingStrategy):
    def __init__(self, k: int = 10, temperature: float = 0.8):
        self.k, self.temperature = k, temperature

    def decode(self, model, input_ids, max_new_tokens, eos_id):
        return top_k_sampling_decode(model, input_ids, max_new_tokens, k=self.k,
                                      temperature=self.temperature, eos_id=eos_id)


class TopP(DecodingStrategy):
    def __init__(self, p: float = 0.9, temperature: float = 0.8):
        self.p, self.temperature = p, temperature

    def decode(self, model, input_ids, max_new_tokens, eos_id):
        return top_p_sampling_decode(model, input_ids, max_new_tokens, p=self.p,
                                      temperature=self.temperature, eos_id=eos_id)


class IndicTokenizer:
    """Thin wrapper around SentencePiece so callers depend on our interface, not the library directly."""
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

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
        ids = torch.tensor([self.tokenizer.encode(text)], dtype=torch.long)
        lengths = torch.tensor([ids.shape[1]])
        with torch.no_grad():
            logits = self.classifier(ids, lengths)
        return torch.argmax(logits, dim=-1).item() == self.accept_label


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

    def generate(self, prompt: str, strategy: DecodingStrategy, max_new_tokens: int = 20,
                 max_filter_attempts: int = 5, seed: int | None = None) -> str:
        prompt_ids = [self.tokenizer.bos_id] + self.tokenizer.encode(prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long)

        for attempt in range(max_filter_attempts):
            if seed is not None:
                torch.manual_seed(seed + attempt)
            out_ids = strategy.decode(self.model, input_ids, max_new_tokens, self.tokenizer.eos_id)
            text = self.tokenizer.decode(out_ids[0].tolist())

            if self.safety_filter is None or self.safety_filter.passes(text):
                return text
        return text  # exhausted attempts; return the last candidate anyway


if __name__ == "__main__":
    ckpt = torch.load("model/tinygpt_multilingual.pt", map_location="cpu")
    model = TinyGPT(vocab_size=ckpt["vocab_size"], d_model=ckpt["d_model"],
                     n_heads=ckpt["n_heads"], n_layers=ckpt["n_layers"],
                     max_seq_len=ckpt["max_seq_len"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tokenizer = IndicTokenizer("tokenizer/multilingual_bpe.model")

    classifier = TextClassifier(vocab_size=tokenizer.vocab_size)
    classifier.load_state_dict(torch.load("classifier/sentiment_classifier.pt", map_location="cpu"))
    classifier.eval()
    safety_filter = SafetyFilter(classifier, tokenizer)

    pipeline = MultilingualGenerationPipeline(model, tokenizer, safety_filter)

    print("OOP pipeline demo -- same interface, swappable decoding strategy:\n")
    for name, strategy in [("Greedy", Greedy()), ("BeamSearch", BeamSearch(beam_width=4)),
                            ("TopK", TopK(k=10)), ("TopP", TopP(p=0.9))]:
        text = pipeline.generate("<te> విద్యార్థి", strategy, max_new_tokens=15, seed=0)
        print(f"  {name:12s}: {text}")
