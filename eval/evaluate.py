"""
Evaluation harness for the tokenizer + language model pipeline.

Metrics:
1. Perplexity on held-out validation data (standard LM quality metric).
2. Tokenizer efficiency: average tokens/word, and unknown-token rate, comparing
   the SentencePiece tokenizer against the from-scratch BPE tokenizer.
3. Self-BLEU across decoding strategies (a common diversity proxy: LOWER self-BLEU
   among generations = MORE diverse output, which is what we'd expect top-p > beam).
"""
import torch
import torch.nn.functional as F
import sentencepiece as spm
from collections import Counter
import math
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.lm import TinyGPT
from model.train import load_data
from decoding.strategies import greedy_decode, beam_search_decode, top_k_sampling_decode, top_p_sampling_decode

SP_MODEL = "tokenizer/multilingual_bpe.model"
CKPT_PATH = "model/tinygpt_multilingual.pt"


def load_model():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = TinyGPT(vocab_size=ckpt["vocab_size"], d_model=ckpt["d_model"],
                     n_heads=ckpt["n_heads"], n_layers=ckpt["n_layers"],
                     max_seq_len=ckpt["max_seq_len"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def eval_perplexity(model, sp):
    _, val_data, _ = load_data(sp)
    seq_len = model.max_seq_len
    losses = []
    with torch.no_grad():
        for i in range(0, len(val_data) - seq_len - 1, seq_len):
            x = val_data[i:i + seq_len].unsqueeze(0)
            y = val_data[i + 1:i + seq_len + 1].unsqueeze(0)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            losses.append(loss.item())
    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(avg_loss)
    print(f"Held-out validation perplexity: {ppl:.2f}  (avg cross-entropy: {avg_loss:.3f})")
    return ppl


def eval_tokenizer_efficiency(sp):
    with open("data/corpus_hi.txt", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()][:200]  # sample for speed

    total_words, total_pieces, unk_count = 0, 0, 0
    for line in lines:
        n_words = len(line.split())
        ids = sp.encode(line, out_type=int)
        total_words += n_words
        total_pieces += len(ids)
        unk_count += sum(1 for i in ids if i == sp.unk_id())

    print(f"\nTokenizer efficiency (SentencePiece BPE, sampled {len(lines)} sentences):")
    print(f"  Avg subword pieces per word : {total_pieces / total_words:.2f}")
    print(f"  Unknown-token rate          : {unk_count / total_pieces * 100:.2f}%")


def ngram_counts(tokens, n):
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def self_bleu_pairwise(sentences_tokens, max_n=2):
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


def eval_decoding_diversity(model, sp):
    prompts = ["<hi> राम", "<hi> बच्चे पार्क में", "<te> విద్యార్థి", "<ml> കുട്ടികൾ", "<hi> दोस्त"]
    eos_id = sp.eos_id()
    strategies = {
        "greedy": lambda ids: greedy_decode(model, ids, 15, eos_id=eos_id),
        "beam (w=4)": lambda ids: beam_search_decode(model, ids, 15, beam_width=4, eos_id=eos_id),
        "top-k (k=10)": lambda ids: top_k_sampling_decode(model, ids, 15, k=10, temperature=0.8, eos_id=eos_id),
        "top-p (p=0.9)": lambda ids: top_p_sampling_decode(model, ids, 15, p=0.9, temperature=0.8, eos_id=eos_id),
    }

    print("\nDecoding diversity (self-BLEU across 5 prompts; LOWER = more diverse output):")
    for name, fn in strategies.items():
        torch.manual_seed(0)
        outputs = []
        for p in prompts:
            ids = torch.tensor([[sp.bos_id()] + sp.encode(p, out_type=int)])
            out = fn(ids)
            outputs.append(out[0].tolist())
        score = self_bleu_pairwise(outputs)
        print(f"  {name:15s}: self-BLEU = {score:.3f}")


if __name__ == "__main__":
    sp = spm.SentencePieceProcessor(model_file=SP_MODEL)
    model = load_model()

    print("=" * 60)
    eval_perplexity(model, sp)
    print("=" * 60)
    eval_tokenizer_efficiency(sp)
    print("=" * 60)
    eval_decoding_diversity(model, sp)
    print("=" * 60)
