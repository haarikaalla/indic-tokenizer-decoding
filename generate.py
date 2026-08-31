"""
CLI comparison of every decoding strategy on the same prompt.

    python generate.py
    python generate.py --prompt "<hi> राम" --max-new-tokens 30 --seed 7

Model and tokenizer are loaded once and reused across prompts, rather than being
reloaded for every call.
"""
from __future__ import annotations

import argparse
import logging

import torch

import config
import loaders
from decoding.strategies import (
    greedy_decode, beam_search_decode, top_k_sampling_decode, top_p_sampling_decode
)

logger = logging.getLogger("generate")

DEMO_PROMPTS = [
    "<hi> राम",
    "<te> విద్యార్థి",
    "<ml> കുട്ടികൾ",
    "<kn> ವಿದ್ಯಾರ್ಥಿ",
]


def run(prompt: str, model, sp, max_new_tokens: int = 20, seed: int | None = None) -> dict[str, str]:
    """Decode `prompt` with all four strategies and return {strategy_name: text}."""
    if not prompt.strip():
        raise ValueError("prompt must be a non-empty string, e.g. '<hi> राम'.")

    device = next(model.parameters()).device
    prompt_ids = [sp.bos_id()] + sp.encode(prompt, out_type=int)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    eos_id = sp.eos_id()

    results: dict[str, str] = {}
    results["Greedy"] = sp.decode(
        greedy_decode(model, input_ids, max_new_tokens, eos_id=eos_id)[0].tolist())
    results["Beam (w=4)"] = sp.decode(
        beam_search_decode(model, input_ids, max_new_tokens, beam_width=4, eos_id=eos_id)[0].tolist())

    if seed is not None:
        torch.manual_seed(seed)
    results["Top-k (k=10)"] = sp.decode(
        top_k_sampling_decode(model, input_ids, max_new_tokens, k=10, temperature=0.8, eos_id=eos_id)[0].tolist())

    if seed is not None:
        torch.manual_seed(seed)
    results["Top-p (p=0.9)"] = sp.decode(
        top_p_sampling_decode(model, input_ids, max_new_tokens, p=0.9, temperature=0.8, eos_id=eos_id)[0].tolist())
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", action="append",
                        help="Language-tagged prompt. Repeatable. Defaults to one demo prompt per language.")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=config.SEED)
    return parser.parse_args()


def main() -> None:
    config.setup_logging()
    args = parse_args()
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be >= 1")

    device = config.resolve_device()
    model = loaders.load_language_model(device=device)
    sp = loaders.load_sentencepiece()

    for prompt in (args.prompt or DEMO_PROMPTS):
        logger.info("Prompt: %r", prompt)
        for name, text in run(prompt, model, sp, args.max_new_tokens, args.seed).items():
            logger.info("  %-14s: %s", name, text)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
