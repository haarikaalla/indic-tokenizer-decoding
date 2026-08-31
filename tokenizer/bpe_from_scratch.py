"""
Byte Pair Encoding, implemented from scratch (no libraries), for interview-defensibility.

Algorithm (Sennrich et al., 2016):
1. Start with a vocabulary of individual characters.
2. Represent every word as a sequence of characters + an end-of-word marker.
3. Count all adjacent symbol pairs across the corpus.
4. Merge the most frequent pair into a new symbol; add it to the vocabulary.
5. Repeat steps 3-4 for `num_merges` iterations.

This mirrors what SentencePiece does internally in BPE mode (train_sentencepiece.py),
but written explicitly so every step is inspectable.
"""
from __future__ import annotations

import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

if __package__ in (None, ""):  # `python tokenizer/bpe_from_scratch.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger("bpe")

END_OF_WORD = "</w>"

Pair = tuple[str, str]
Word = tuple[str, ...]


class SimpleBPE:
    """Byte Pair Encoding trained on whitespace-tokenized text."""

    def __init__(self, num_merges: int = 150, min_pair_freq: int = 2) -> None:
        if num_merges < 0:
            raise ValueError(f"num_merges must be >= 0, got {num_merges}.")
        if min_pair_freq < 1:
            raise ValueError(f"min_pair_freq must be >= 1, got {min_pair_freq}.")
        self.num_merges = num_merges
        self.min_pair_freq = min_pair_freq
        self.merges: list[Pair] = []   # ordered list of pairs; order defines merge priority
        self.vocab: set[str] = set()

    def _get_word_freqs(self, corpus_lines: list[str]) -> Counter[Word]:
        """Split each line into whitespace-separated words, count frequencies.
        Each word is represented as a tuple of characters + end-of-word marker."""
        word_freqs: Counter[Word] = Counter()
        for line in corpus_lines:
            for word in line.strip().split():
                chars = tuple(word) + (END_OF_WORD,)
                word_freqs[chars] += 1
        return word_freqs

    def _get_pair_counts(self, word_freqs: Counter[Word] | dict[Word, int]) -> dict[Pair, int]:
        """Count every adjacent symbol pair across the corpus, weighted by word frequency."""
        pairs: dict[Pair, int] = defaultdict(int)
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def _merge_pair(self, pair: Pair, word_freqs: dict[Word, int]) -> dict[Word, int]:
        """Replace every occurrence of `pair` with its merged symbol in every word."""
        new_word_freqs: dict[Word, int] = {}
        bigram = pair[0] + pair[1]
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                    new_word.append(bigram)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        return new_word_freqs

    def train(self, corpus_lines: list[str]) -> None:
        """Learn up to `num_merges` merges from the corpus, stopping when no pair is frequent enough."""
        word_freqs: dict[Word, int] = dict(self._get_word_freqs(corpus_lines))
        # initial vocab = all individual characters seen
        for word in word_freqs:
            self.vocab.update(word)

        for step in range(self.num_merges):
            pairs = self._get_pair_counts(word_freqs)
            if not pairs:
                break
            best_pair = max(pairs, key=lambda key: pairs[key])
            if pairs[best_pair] < self.min_pair_freq:
                break  # no more useful merges
            word_freqs = self._merge_pair(best_pair, word_freqs)
            self.merges.append(best_pair)
            self.vocab.add(best_pair[0] + best_pair[1])
            if step < 10 or step % 25 == 0:
                logger.debug("merge %3d: %s -> %r (freq=%d)",
                             step, best_pair, best_pair[0] + best_pair[1], pairs[best_pair])

    def tokenize_word(self, word: str) -> list[str]:
        """Apply learned merges, in order, to a single word."""
        symbols = list(word) + [END_OF_WORD]
        for pair in self.merges:
            i = 0
            new_symbols = []
            bigram = pair[0] + pair[1]
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                    new_symbols.append(bigram)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    def tokenize(self, text: str) -> list[str]:
        """Tokenize a whole line into subword symbols."""
        tokens: list[str] = []
        for word in text.strip().split():
            tokens.extend(self.tokenize_word(word))
        return tokens


def main() -> None:
    import config

    config.setup_logging()
    corpus = config.require_file(
        config.LANG_CORPUS["hi"], "Generate the corpora first: python data/generate_corpus.py"
    )
    with corpus.open(encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    if not lines:
        raise ValueError(f"Corpus {corpus} is empty; nothing to train on.")

    bpe = SimpleBPE(num_merges=150)
    logger.info("Training from-scratch BPE on %d sentences...", len(lines))
    bpe.train(lines)

    logger.info("Final vocab size: %d", len(bpe.vocab))
    logger.info("Number of merges learned: %d", len(bpe.merges))

    for sample in ["राम स्कूल जाता है।", "बच्चे पार्क में खेलते हैं।"]:
        logger.info("Input : %s", sample)
        logger.info("Tokens: %s", bpe.tokenize(sample))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
