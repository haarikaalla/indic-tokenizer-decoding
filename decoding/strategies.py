"""
Text decoding strategies, implemented from scratch on raw model logits.
No calls to model.generate() anywhere -- every strategy manually manages
the autoregressive loop, so the mechanics are fully inspectable.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def greedy_decode(model, input_ids, max_new_tokens, eos_id=None):
    """At every step, pick the single highest-probability next token."""
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
def beam_search_decode(model, input_ids, max_new_tokens, beam_width=4, eos_id=None):
    """
    Maintain `beam_width` candidate sequences, ranked by cumulative log-probability.
    At each step, expand every beam by every possible next token, then keep only
    the top `beam_width` sequences overall.
    """
    device = input_ids.device
    # each beam: (sequence tensor, cumulative log-prob, finished flag)
    beams = [(input_ids.clone(), 0.0, False)]

    for _ in range(max_new_tokens):
        candidates = []
        for seq, score, finished in beams:
            if finished:
                candidates.append((seq, score, finished))
                continue
            idx_cond = seq[:, -model.max_seq_len:]
            logits = model(idx_cond)[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # (vocab,)

            topk_log_probs, topk_ids = torch.topk(log_probs, beam_width)
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
def top_k_sampling_decode(model, input_ids, max_new_tokens, k=10, temperature=1.0, eos_id=None):
    """At each step, restrict sampling to the top-k highest-probability tokens."""
    idx = input_ids.clone()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.max_seq_len:]
        logits = model(idx_cond)[:, -1, :] / temperature
        topk_vals, topk_idx = torch.topk(logits, k)
        probs = F.softmax(topk_vals, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)          # index into topk_idx
        next_id = topk_idx.gather(-1, sampled)
        idx = torch.cat([idx, next_id], dim=1)
        if eos_id is not None and next_id.item() == eos_id:
            break
    return idx


@torch.no_grad()
def top_p_sampling_decode(model, input_ids, max_new_tokens, p=0.9, temperature=1.0, eos_id=None):
    """
    Nucleus sampling: sort tokens by probability, keep the smallest set whose
    cumulative probability exceeds p, renormalize, then sample from that set.
    """
    idx = input_ids.clone()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.max_seq_len:]
        logits = model(idx_cond)[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)

        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)

        # keep smallest nucleus with cumulative prob > p (always keep at least 1 token)
        cutoff = (cumulative > p).float().argmax(dim=-1).item()
        keep = max(cutoff + 1, 1)

        nucleus_probs = sorted_probs[:, :keep]
        nucleus_idx = sorted_idx[:, :keep]
        nucleus_probs = nucleus_probs / nucleus_probs.sum(dim=-1, keepdim=True)

        sampled = torch.multinomial(nucleus_probs, num_samples=1)
        next_id = nucleus_idx.gather(-1, sampled)
        idx = torch.cat([idx, next_id], dim=1)
        if eos_id is not None and next_id.item() == eos_id:
            break
    return idx
