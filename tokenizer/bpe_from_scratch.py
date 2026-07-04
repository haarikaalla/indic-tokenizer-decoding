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
from collections import defaultdict, Counter
import re

END_OF_WORD = "</w>"

class SimpleBPE:
    def __init__(self, num_merges=150):
        self.num_merges = num_merges
        self.merges = []          # ordered list of (pair) -> merged, defines merge priority
        self.vocab = set()

    def _get_word_freqs(self, corpus_lines):
        """Split each line into whitespace-separated words, count frequencies.
        Each word is represented as a tuple of characters + end-of-word marker."""
        word_freqs = Counter()
        for line in corpus_lines:
            for word in line.strip().split():
                chars = tuple(word) + (END_OF_WORD,)
                word_freqs[chars] += 1
        return word_freqs

    def _get_pair_counts(self, word_freqs):
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def _merge_pair(self, pair, word_freqs):
        new_word_freqs = {}
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

    def train(self, corpus_lines):
        word_freqs = self._get_word_freqs(corpus_lines)
        # initial vocab = all individual characters seen
        for word in word_freqs:
            self.vocab.update(word)

        for step in range(self.num_merges):
            pairs = self._get_pair_counts(word_freqs)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < 2:
                break  # no more useful merges
            word_freqs = self._merge_pair(best_pair, word_freqs)
            self.merges.append(best_pair)
            self.vocab.add(best_pair[0] + best_pair[1])
            if step < 10 or step % 25 == 0:
                print(f"merge {step:3d}: {best_pair} -> '{best_pair[0]+best_pair[1]}' "
                      f"(freq={pairs[best_pair]})")

    def tokenize_word(self, word):
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

    def tokenize(self, text):
        tokens = []
        for word in text.strip().split():
            tokens.extend(self.tokenize_word(word))
        return tokens


if __name__ == "__main__":
    with open("data/corpus_hi.txt", encoding="utf-8") as f:
        lines = f.readlines()

    bpe = SimpleBPE(num_merges=150)
    print(f"Training from-scratch BPE on {len(lines)} sentences...\n")
    bpe.train(lines)

    print(f"\nFinal vocab size: {len(bpe.vocab)}")
    print(f"Number of merges learned: {len(bpe.merges)}")

    samples = ["राम स्कूल जाता है।", "बच्चे पार्क में खेलते हैं।"]
    for s in samples:
        print(f"\nInput : {s}")
        print(f"Tokens: {bpe.tokenize(s)}")
