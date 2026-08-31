"""
Tests for the from-scratch decoding strategies (decoding/strategies.py).

Uses a tiny, untrained TinyGPT purely as a shape/contract fixture -- these tests
check decoding MECHANICS (output shape, length bounds, EOS handling, determinism),
not generation quality, so they run in well under a second with no training needed.

Run: pytest tests/ -v
"""
import torch
import pytest

from model.lm import TinyGPT
from decoding.strategies import (
    greedy_decode, beam_search_decode, top_k_sampling_decode, top_p_sampling_decode,
)

VOCAB_SIZE = 50
MAX_SEQ_LEN = 16


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    model = TinyGPT(vocab_size=VOCAB_SIZE, d_model=16, n_heads=2, n_layers=1, max_seq_len=MAX_SEQ_LEN)
    model.eval()
    return model


@pytest.fixture
def prompt_ids():
    return torch.tensor([[1, 2, 3]], dtype=torch.long)  # batch of 1, 3 tokens


def test_greedy_decode_extends_sequence(tiny_model, prompt_ids):
    out = greedy_decode(tiny_model, prompt_ids, max_new_tokens=5)
    assert out.shape[0] == 1
    assert out.shape[1] <= prompt_ids.shape[1] + 5
    assert out.shape[1] >= prompt_ids.shape[1]
    assert torch.equal(out[:, :prompt_ids.shape[1]], prompt_ids), \
        "Decoded sequence must start with the original prompt, unmodified"


def test_greedy_decode_is_deterministic(tiny_model, prompt_ids):
    out1 = greedy_decode(tiny_model, prompt_ids, max_new_tokens=5)
    out2 = greedy_decode(tiny_model, prompt_ids, max_new_tokens=5)
    assert torch.equal(out1, out2), "Greedy decoding must be fully deterministic (argmax, no sampling)"


def test_greedy_decode_stops_at_eos(tiny_model, prompt_ids):
    """If eos_id happens to be the very next argmax pick, generation should stop early."""
    with torch.no_grad():
        logits = tiny_model(prompt_ids)[:, -1, :]
    forced_eos = torch.argmax(logits, dim=-1).item()
    out = greedy_decode(tiny_model, prompt_ids, max_new_tokens=10, eos_id=forced_eos)
    assert out.shape[1] == prompt_ids.shape[1] + 1, \
        "Generation should stop the step after EOS is produced, not run all max_new_tokens"


def test_beam_search_respects_beam_width_and_length(tiny_model, prompt_ids):
    out = beam_search_decode(tiny_model, prompt_ids, max_new_tokens=5, beam_width=3)
    assert out.shape[0] == 1  # returns the single best beam
    assert out.shape[1] <= prompt_ids.shape[1] + 5


def test_top_k_sampling_is_seed_reproducible(tiny_model, prompt_ids):
    torch.manual_seed(42)
    out1 = top_k_sampling_decode(tiny_model, prompt_ids, max_new_tokens=5, k=5, temperature=0.8)
    torch.manual_seed(42)
    out2 = top_k_sampling_decode(tiny_model, prompt_ids, max_new_tokens=5, k=5, temperature=0.8)
    assert torch.equal(out1, out2), "Same seed must produce identical sampled output"


def test_top_p_sampling_is_seed_reproducible(tiny_model, prompt_ids):
    torch.manual_seed(7)
    out1 = top_p_sampling_decode(tiny_model, prompt_ids, max_new_tokens=5, p=0.9, temperature=0.8)
    torch.manual_seed(7)
    out2 = top_p_sampling_decode(tiny_model, prompt_ids, max_new_tokens=5, p=0.9, temperature=0.8)
    assert torch.equal(out1, out2), "Same seed must produce identical sampled output"


def test_all_strategies_never_exceed_model_context_window(tiny_model, prompt_ids):
    """Regression guard: idx[:, -max_seq_len:] slicing in each strategy must never
    let the model be called with a sequence longer than it supports."""
    for decode_fn, kwargs in [
        (greedy_decode, {}),
        (beam_search_decode, {"beam_width": 2}),
        (top_k_sampling_decode, {"k": 5}),
        (top_p_sampling_decode, {"p": 0.9}),
    ]:
        out = decode_fn(tiny_model, prompt_ids, max_new_tokens=30, **kwargs)
        assert out.shape[1] <= prompt_ids.shape[1] + 30


# --- Input validation: bad arguments must fail loudly, not silently misbehave ---

@pytest.mark.parametrize("decode_fn,kwargs", [
    (greedy_decode, {}),
    (beam_search_decode, {"beam_width": 2}),
    (top_k_sampling_decode, {"k": 5}),
    (top_p_sampling_decode, {"p": 0.9}),
])
def test_batched_input_is_rejected(tiny_model, decode_fn, kwargs):
    """These implementations assume batch size 1; a silent wrong answer would be worse."""
    batched = torch.randint(0, VOCAB_SIZE, (2, 4))
    with pytest.raises(ValueError):
        decode_fn(tiny_model, batched, max_new_tokens=3, **kwargs)


@pytest.mark.parametrize("decode_fn,kwargs", [
    (greedy_decode, {}),
    (beam_search_decode, {"beam_width": 2}),
    (top_k_sampling_decode, {"k": 5}),
    (top_p_sampling_decode, {"p": 0.9}),
])
def test_one_dimensional_input_is_rejected(tiny_model, decode_fn, kwargs):
    with pytest.raises(ValueError):
        decode_fn(tiny_model, torch.tensor([1, 2, 3]), max_new_tokens=3, **kwargs)


@pytest.mark.parametrize("decode_fn,kwargs", [
    (top_k_sampling_decode, {"k": 5}),
    (top_p_sampling_decode, {"p": 0.9}),
])
def test_nonpositive_temperature_is_rejected(tiny_model, prompt_ids, decode_fn, kwargs):
    with pytest.raises(ValueError):
        decode_fn(tiny_model, prompt_ids, max_new_tokens=3, temperature=0.0, **kwargs)


def test_k_larger_than_vocab_is_clamped_not_crashed(tiny_model, prompt_ids):
    """torch.topk would raise if k exceeded the vocabulary; we clamp instead."""
    out = top_k_sampling_decode(tiny_model, prompt_ids, max_new_tokens=3, k=VOCAB_SIZE * 10)
    assert out.shape[1] == prompt_ids.shape[1] + 3


def test_beam_width_larger_than_vocab_is_clamped(tiny_model, prompt_ids):
    out = beam_search_decode(tiny_model, prompt_ids, max_new_tokens=3, beam_width=VOCAB_SIZE * 10)
    assert out.shape[1] <= prompt_ids.shape[1] + 3


def test_top_p_of_one_keeps_the_full_distribution(tiny_model, prompt_ids):
    """p=1.0 must not collapse the nucleus to a single token (a real bug we fixed)."""
    out = top_p_sampling_decode(tiny_model, prompt_ids, max_new_tokens=5, p=1.0)
    assert out.shape[1] == prompt_ids.shape[1] + 5


@pytest.mark.parametrize("bad_p", [0.0, -0.5, 1.5])
def test_out_of_range_p_is_rejected(tiny_model, prompt_ids, bad_p):
    with pytest.raises(ValueError):
        top_p_sampling_decode(tiny_model, prompt_ids, max_new_tokens=3, p=bad_p)


@pytest.mark.parametrize("bad_k", [0, -3])
def test_nonpositive_k_is_rejected(tiny_model, prompt_ids, bad_k):
    with pytest.raises(ValueError):
        top_k_sampling_decode(tiny_model, prompt_ids, max_new_tokens=3, k=bad_k)


def test_model_raises_a_clear_error_beyond_its_context_window(tiny_model):
    too_long = torch.randint(0, VOCAB_SIZE, (1, MAX_SEQ_LEN + 1))
    with pytest.raises(ValueError, match="context window"):
        tiny_model(too_long)
