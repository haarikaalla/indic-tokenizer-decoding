"""
Safety-controlled generation: generate candidates, then use the trained sentiment
classifier to filter out ones that don't meet a policy (here: reject negative-
sentiment output). This is a simplified but real instance of the "safety-controlled
text composition" pattern -- generate, score, filter/regenerate.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import sentencepiece as spm
import torch
import torch.nn as nn

if __package__ in (None, ""):  # `python classifier/safety_filter_demo.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import loaders
from decoding.strategies import top_p_sampling_decode

logger = logging.getLogger("safety_filter")


@torch.no_grad()
def classify_sentiment(clf: nn.Module, sp: spm.SentencePieceProcessor, text: str) -> tuple[str, float]:
    """Return (label, P(positive)) for a candidate generation."""
    encoded = sp.encode(text, out_type=int)
    if not encoded:
        return "negative", 0.0  # unscoreable text fails closed
    device = next(clf.parameters()).device
    ids = torch.tensor([encoded], dtype=torch.long, device=device)
    lengths = torch.tensor([ids.shape[1]])
    probs = torch.softmax(clf(ids, lengths), dim=-1)
    label = "positive" if int(torch.argmax(probs)) == 1 else "negative"
    return label, probs[0, 1].item()


def generate_with_safety_filter(
    prompt: str,
    lm: nn.Module,
    clf: nn.Module,
    sp: spm.SentencePieceProcessor,
    max_attempts: int = 8,
    max_new_tokens: int = 15,
    seed: int = config.SEED,
) -> str | None:
    """
    Generate, score, and retry until a candidate passes the policy.

    Attempt `n` uses `seed + n`, so the demo is reproducible run to run while each
    retry still explores a different sample.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}.")

    device = next(lm.parameters()).device
    eos_id = sp.eos_id()
    prompt_ids = [sp.bos_id()] + sp.encode(prompt, out_type=int)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    for attempt in range(1, max_attempts + 1):
        torch.manual_seed(seed + attempt)
        out = top_p_sampling_decode(lm, input_ids, max_new_tokens, p=0.9, temperature=0.9, eos_id=eos_id)
        text = sp.decode(out[0].tolist())
        label, pos_score = classify_sentiment(clf, sp, text)
        status = "ACCEPTED" if label == "positive" else "rejected"
        logger.info("  attempt %d: [%s] (P(positive)=%.2f)  %s", attempt, status, pos_score, text)
        if label == "positive":
            return text

    logger.warning("No candidate passed the filter after %d attempts.", max_attempts)
    return None  # exhausted attempts without a passing generation


def main() -> None:
    config.setup_logging()
    device = config.resolve_device()

    sp = loaders.load_sentencepiece()
    lm = loaders.load_language_model(device=device)
    clf = loaders.load_classifier(sp.get_piece_size(), device=device)

    for prompt in ["<hi> राम", "<hi> छात्र"]:
        logger.info("Prompt: %r -- generating until a positive-sentiment output passes the filter", prompt)
        result = generate_with_safety_filter(prompt, lm, clf, sp)
        logger.info("  Final accepted output: %s", result)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
