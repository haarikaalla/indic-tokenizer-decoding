"""
Trains a subword (BPE) tokenizer on the Hindi corpus using SentencePiece.

This is the "use the production tool correctly" half of the tokenizer story.
See bpe_from_scratch.py for the "implement the algorithm myself" half.
"""
import sentencepiece as spm
import os

CORPUS_PATH = "data/corpus_hi.txt"
MODEL_PREFIX = "tokenizer/hindi_bpe"
VOCAB_SIZE = 400  # small on purpose: our toy corpus has a limited vocabulary

def train():
    assert os.path.exists(CORPUS_PATH), f"Corpus not found at {CORPUS_PATH}"
    spm.SentencePieceTrainer.train(
        input=CORPUS_PATH,
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,       # must be 1.0 for non-Latin scripts like Devanagari
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>",
    )
    print(f"Trained tokenizer -> {MODEL_PREFIX}.model / .vocab")

def demo():
    sp = spm.SentencePieceProcessor(model_file=f"{MODEL_PREFIX}.model")
    samples = [
        "राम स्कूल जाता है।",
        "बच्चे पार्क में खेलते हैं और गाना गाते हैं।",
    ]
    for s in samples:
        pieces = sp.encode(s, out_type=str)
        ids = sp.encode(s, out_type=int)
        print(f"\nInput      : {s}")
        print(f"Pieces     : {pieces}")
        print(f"IDs        : {ids}")
        print(f"Decoded    : {sp.decode(ids)}")

if __name__ == "__main__":
    train()
    demo()
