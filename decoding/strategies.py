"""
Text decoding strategies, implemented from scratch on raw model logits.
No calls to model.generate() anywhere -- every strategy manually manages
the autoregressive loop, so the mechanics are fully inspectable.

All strategies operate on a single sequence (batch size 1). That constraint is
validated up front rather than surfacing later as an opaque
``RuntimeError: a Tensor with N elements cannot be converted to Scalar`` from the
``.item()`` call used for EOS detection.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate_inputs(model: torch.nn.Module, input_ids: torch.Tensor, max_new_tokens: int) -> None:
    """Reject the input shapes/arguments that would otherwise fail deep in the loop."""
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be 2-D (batch, seq_len), got shape {tuple(input_ids.shape)}.")
    if input_ids.size(0) != 1:
        raise ValueError(
            f"Decoding strategies support batch size 1, got {input_ids.size(0)}. "
            "Loop over prompts, or batch at a higher level."
        )
    if input_ids.size(1) == 0:
        raise ValueError("input_ids must contain at least one token (a BOS token is enough).")
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}.")
    if not hasattr(model, "max_seq_len"):
        raise AttributeError("model must expose `max_seq_len` so the context window can be respected.")


def _check_temperature(temperature: float) -> float:
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}. Use greedy_decode for argmax.")
    return temperature


@torch.no_grad()
def greedy_decode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_id: int | None = None,
) -> torch.Tensor:
    """At every step, pick the single highest-probability next token."""
    _validate_inputs(model, input_ids, max_new_tokens)
    idx = input_ids.clone()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.max_seq_len:]
        logits = model(idx_cond)[:, -1, :]           # logits for the next token
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
        if eos_id is not None and next_id.item() == eos_id:
            break
    return idx


@torch.no_grad()
def beam_search_decode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    beam_width: int = 4,
    eos_id: int | None = None,
) -> torch.Tensor:
    """
    Maintain `beam_width` candidate sequences, ranked by cumulative log-probability.
    At each step, expand every beam by every possible next token, then keep only
    the top `beam_width` sequences overall.
    """
    _validate_inputs(model, input_ids, max_new_tokens)
    if beam_width < 1:
        raise ValueError(f"beam_width must be >= 1, got {beam_width}.")

    device = input_ids.device
    # each beam: (sequence tensor, cumulative log-prob, finished flag)
    beams: list[tuple[torch.Tensor, float, bool]] = [(input_ids.clone(), 0.0, False)]

    for _ in range(max_new_tokens):
        candidates: list[tuple[torch.Tensor, float, bool]] = []
        for seq, score, finished in beams:
            if finished:
                candidates.append((seq, score, finished))
                continue
            idx_cond = seq[:, -model.max_seq_len:]
            logits = model(idx_cond)[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # (vocab,)

            # never ask for more candidates than the vocabulary actually has
            effective_width = min(beam_width, log_probs.size(-1))
            topk_log_probs, topk_ids = torch.topk(log_probs, effective_width)
            for lp, tok_id in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                new_seq = torch.cat([seq, torch.tensor([[tok_id]], device=device)], dim=1)
                new_finished = (eos_id is not None and tok_id == eos_id)
                candidates.append((new_seq, score + lp, new_finished))

        # keep only the best `beam_width` candidates, by cumulative log-prob
        candidates.sort(key=lambda c: c[1], reverse=True)
        beams = candidates[:beam_width]

        if all(f for _, _, f in beams):
            break

    beams.sort(key=lambda c: c[1], reverse=True)
    return beams[0][0]  # sequence of the single best beam


@torch.no_grad()
def top_k_sampling_decode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    k: int = 10,
    temperature: float = 1.0,
    eos_id: int | None = None,
) -> torch.Tensor:
    """At each step, restrict sampling to the top-k highest-probability tokens."""
    _validate_inputs(model, input_ids, max_new_tokens)
    _check_temperature(temperature)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")

    idx = input_ids.clone()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.max_seq_len:]
        logits = model(idx_cond)[:, -1, :] / temperature
        effective_k = min(k, logits.size(-1))       # k may exceed the vocabulary
        topk_vals, topk_idx = torch.topk(logits, effective_k)
        probs = F.softmax(topk_vals, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)          # index into topk_idx
        next_id = topk_idx.gather(-1, sampled)
        idx = torch.cat([idx, next_id], dim=1)
        if eos_id is not None and next_id.item() == eos_id:
            break
    return idx


@torch.no_grad()
def top_p_sampling_decode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    p: float = 0.9,
    temperature: float = 1.0,
    eos_id: int | None = None,
) -> torch.Tensor:
    """
    Nucleus sampling: sort tokens by probability, keep the smallest set whose
    cumulative probability exceeds p, renormalize, then sample from that set.
    """
    _validate_inputs(model, input_ids, max_new_tokens)
    _check_temperature(temperature)
    if not 0.0 < p <= 1.0:
        raise ValueError(f"p must be in (0, 1], got {p}.")

    idx = input_ids.clone()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.max_seq_len:]
        logits = model(idx_cond)[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)

        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)

        # keep smallest nucleus with cumulative prob > p (always keep at least 1 token).
        # When p == 1.0 (or floating-point drift means nothing ever exceeds p) the
        # mask is all-False and argmax would wrongly collapse the nucleus to one
        # token -- in that case the whole distribution is the nucleus.
        above_p = cumulative > p
        if bool(above_p.any()):
            cutoff = int(above_p.float().argmax(dim=-1).item())
            keep = max(cutoff + 1, 1)
        else:
            keep = sorted_probs.size(-1)

        nucleus_probs = sorted_probs[:, :keep]
        nucleus_idx = sorted_idx[:, :keep]
        denom = nucleus_probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(nucleus_probs.dtype).tiny)
        nucleus_probs = nucleus_probs / denom

        sampled = torch.multinomial(nucleus_probs, num_samples=1)
        next_id = nucleus_idx.gather(-1, sampled)
        idx = torch.cat([idx, next_id], dim=1)
        if eos_id is not None and next_id.item() == eos_id:
            break
    return idx
