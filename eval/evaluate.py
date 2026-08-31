"""
Evaluation harness for the tokenizer + language model pipeline.

Metrics:
1. Perplexity on held-out validation data (standard LM quality metric).
2. Tokenizer efficiency: average tokens/word, and unknown-token rate, comparing
   the SentencePiece tokenizer against the from-scratch BPE tokenizer.
3. Self-BLEU across decoding strategies (a common diversity proxy: LOWER self-BLEU
   among generations = MORE diverse output, which is what we'd expect top-p > beam).
"""
from __future__ import annotations

import logging
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

import sentencepiece as spm
import torch
import torch.nn.functional as F

if __package__ in (None, ""):  # `python eval/evaluate.py` -- put the repo root on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import loaders
from decoding.strategies import greedy_decode, beam_search_decode, top_k_sampling_decode, top_p_sampling_decode
from model.lm import TinyGPT
from model.train import load_data

logger = logging.getLogger("eval")


def eval_perplexity(model: TinyGPT, sp: spm.SentencePieceProcessor) -> float:
    """Perplexity over non-overlapping validation windows (standard LM quality metric)."""
    _, val_data, _ = load_data(sp)
    seq_len = model.max_seq_len
    if len(val_data) < seq_len + 2:
        raise ValueError(
            f"Validation split has {len(val_data)} tokens, too few for a {seq_len}-token window."
        )

    device = next(model.parameters()).device
    losses = []
    with torch.no_grad():
        for i in range(0, len(val_data) - seq_len - 1, seq_len):
            x = val_data[i:i + seq_len].unsqueeze(0).to(device)
            y = val_data[i + 1:i + seq_len + 1].unsqueeze(0).to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(min(avg_loss, 700))  # exp overflows past ~709
    logger.info("Held-out validation perplexity: %.2f  (avg cross-entropy: %.3f)", ppl, avg_loss)
    return ppl


def eval_tokenizer_efficiency(sp: spm.SentencePieceProcessor, corpus_path=None, sample: int = 200) -> None:
    """Report average subword pieces per word and the unknown-token rate."""
    path = config.require_file(
        corpus_path or config.LANG_CORPUS["hi"],
        "Generate the corpora first: python data/generate_corpus.py",
    )
    with path.open(encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()][:sample]
    if not lines:
        raise ValueError(f"Corpus {path} is empty.")

    total_words, total_pieces, unk_count = 0, 0, 0
    for line in lines:
        ids = sp.encode(line, out_type=int)
        total_words += len(line.split())
        total_pieces += len(ids)
        unk_count += sum(1 for i in ids if i == sp.unk_id())

    if total_words == 0 or total_pieces == 0:
        logger.warning("Sampled corpus produced no words/pieces; skipping efficiency report.")
        return

    logger.info("Tokenizer efficiency (SentencePiece BPE, sampled %d sentences):", len(lines))
    logger.info("  Avg subword pieces per word : %.2f", total_pieces / total_words)
    logger.info("  Unknown-token rate          : %.2f%%", unk_count / total_pieces * 100)


def ngram_counts(tokens: Sequence[int], n: int) -> Counter:
    """Count the n-grams in a token sequence."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def self_bleu_pairwise(sentences_tokens: Sequence[Sequence[int]], max_n: int = 2) -> float:
    """Simplified self-BLEU: average pairwise n-gram overlap across a set of generations.
    Lower = more diverse (independent generations look less like each other)."""
    if len(sentences_tokens) < 2:
        return 0.0
    scores = []
    for i in range(len(sentences_tokens)):
        for j in range(len(sentences_tokens)):
            if i == j:
                continue
            hyp, ref = sentences_tokens[i], sentences_tokens[j]
            precisions = []
            for n in range(1, max_n + 1):
                hyp_ngrams = ngram_counts(hyp, n)
                ref_ngrams = ngram_counts(ref, n)
                overlap = sum((hyp_ngrams & ref_ngrams).values())
                total = max(sum(hyp_ngrams.values()), 1)
                precisions.append(overlap / total)
            scores.append(sum(precisions) / len(precisions))
    return sum(scores) / len(scores)


def eval_decoding_diversity(model: TinyGPT, sp: spm.SentencePieceProcessor) -> dict[str, float]:
    """Compare how much each decoding strategy repeats itself across prompts."""
    prompts = ["<hi> राम", "<hi> बच्चे पार्क में", "<te> విద్యార్థి", "<ml> കുട്ടികൾ",
               "<kn> ವಿದ್ಯಾರ್ಥಿ", "<hi> दोस्त"]
    eos_id = sp.eos_id()
    device = next(model.parameters()).device
    strategies = {
        "greedy": lambda ids: greedy_decode(model, ids, 15, eos_id=eos_id),
        "beam (w=4)": lambda ids: beam_search_decode(model, ids, 15, beam_width=4, eos_id=eos_id),
        "top-k (k=10)": lambda ids: top_k_sampling_decode(model, ids, 15, k=10, temperature=0.8, eos_id=eos_id),
        "top-p (p=0.9)": lambda ids: top_p_sampling_decode(model, ids, 15, p=0.9, temperature=0.8, eos_id=eos_id),
    }

    logger.info("Decoding diversity (self-BLEU across %d prompts; LOWER = more diverse):", len(prompts))
    results: dict[str, float] = {}
    for name, fn in strategies.items():
        torch.manual_seed(config.SEED)
        outputs = []
        for prompt in prompts:
            ids = torch.tensor([[sp.bos_id()] + sp.encode(prompt, out_type=int)], device=device)
            outputs.append(fn(ids)[0].tolist())
        results[name] = self_bleu_pairwise(outputs)
        logger.info("  %-15s: self-BLEU = %.3f", name, results[name])
    return results


def main() -> None:
    config.setup_logging()
    config.seed_everything()
    device = config.resolve_device()

    sp = loaders.load_sentencepiece()
    model = loaders.load_language_model(device=device)

    eval_perplexity(model, sp)
    eval_tokenizer_efficiency(sp)
    eval_decoding_diversity(model, sp)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
