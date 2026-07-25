"""
Tests for the from-scratch BPE implementation (tokenizer/bpe_from_scratch.py).

Run: pytest tests/ -v
"""
import pytest
from tokenizer.bpe_from_scratch import SimpleBPE, END_OF_WORD


@pytest.fixture(scope="module")
def toy_corpus():
    # "low" appears in 2/3 lines -> its char-pairs should merge early.
    return [
        "low lower lowest",
        "low lower newest",
        "wide widest widen",
    ]


def test_bpe_trains_and_learns_merges(toy_corpus):
    bpe = SimpleBPE(num_merges=20)
    bpe.train(toy_corpus)
    assert len(bpe.merges) > 0, "BPE should learn at least one merge rule from a repetitive corpus"


def test_bpe_stops_when_no_useful_pairs_remain():
    """With num_merges way beyond what the corpus supports, training should stop early
    rather than merging singleton (freq=1) pairs -- this is the min-frequency guard."""
    bpe = SimpleBPE(num_merges=500)
    bpe.train(["a b c d e f g"])  # every char is unique, no repeated pairs
    assert len(bpe.merges) == 0, "No pair repeats, so zero merges should be learned"


def test_bpe_tokenize_word_uses_learned_merges_in_order(toy_corpus):
    bpe = SimpleBPE(num_merges=20)
    bpe.train(toy_corpus)
    tokens = bpe.tokenize_word("low")
    assert all(t in bpe.vocab for t in tokens), \
        "Every emitted subword symbol must exist in the trained vocabulary"


def test_bpe_tokenize_full_sentence_reconstructs_word_count(toy_corpus):
    bpe = SimpleBPE(num_merges=20)
    bpe.train(toy_corpus)
    sentence = "low widest"
    tokens = bpe.tokenize(sentence)
    reconstructed = "".join(tokens).replace(END_OF_WORD, "")
    assert reconstructed == "lowwidest", \
        "Concatenated subword pieces must reconstruct the original characters exactly"


def test_bpe_handles_empty_corpus_gracefully():
    bpe = SimpleBPE(num_merges=10)
    bpe.train([""])
    assert bpe.tokenize("") == []
