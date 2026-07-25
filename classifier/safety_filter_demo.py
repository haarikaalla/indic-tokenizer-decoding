"""
Safety-controlled generation: generate candidates, then use the trained sentiment
classifier to filter out ones that don't meet a policy (here: reject negative-
sentiment output). This is a simplified but real instance of the "safety-controlled
text composition" pattern -- generate, score, filter/regenerate.
"""
import torch
import sentencepiece as spm
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.lm import TinyGPT
from classifier.classifier_model import TextClassifier
from decoding.strategies import top_p_sampling_decode

SP_MODEL = "tokenizer/multilingual_bpe.model"
LM_CKPT = "model/tinygpt_multilingual.pt"
CLS_CKPT = "classifier/sentiment_classifier.pt"

def load_lm():
    ckpt = torch.load(LM_CKPT, map_location="cpu")
    model = TinyGPT(vocab_size=ckpt["vocab_size"], d_model=ckpt["d_model"],
                     n_heads=ckpt["n_heads"], n_layers=ckpt["n_layers"],
                     max_seq_len=ckpt["max_seq_len"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def load_classifier(vocab_size):
    model = TextClassifier(vocab_size=vocab_size)
    model.load_state_dict(torch.load(CLS_CKPT, map_location="cpu"))
    model.eval()
    return model

def classify_sentiment(clf, sp, text):
    ids = torch.tensor([sp.encode(text, out_type=int)], dtype=torch.long)
    lengths = torch.tensor([ids.shape[1]])
    with torch.no_grad():
        logits = clf(ids, lengths)
        probs = torch.softmax(logits, dim=-1)
    label = "positive" if torch.argmax(probs) == 1 else "negative"
    return label, probs[0, 1].item()  # (label, P(positive))

def generate_with_safety_filter(prompt, lm, clf, sp, max_attempts=8, max_new_tokens=15):
    eos_id = sp.eos_id()
    prompt_ids = [sp.bos_id()] + sp.encode(prompt, out_type=int)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)

    for attempt in range(1, max_attempts + 1):
        torch.manual_seed(attempt)
        out = top_p_sampling_decode(lm, input_ids, max_new_tokens, p=0.9, temperature=0.9, eos_id=eos_id)
        text = sp.decode(out[0].tolist())
        label, pos_score = classify_sentiment(clf, sp, text)
        status = "ACCEPTED" if label == "positive" else "rejected"
        print(f"  attempt {attempt}: [{status}] (P(positive)={pos_score:.2f})  {text}")
        if label == "positive":
            return text
    return None  # exhausted attempts without a passing generation

if __name__ == "__main__":
    sp = spm.SentencePieceProcessor(model_file=SP_MODEL)
    lm = load_lm()
    clf = load_classifier(sp.get_piece_size())

    for prompt in ["<hi> राम", "<hi> छात्र"]:
        print(f"\nPrompt: {prompt!r} -- generating until a positive-sentiment output passes the filter")
        result = generate_with_safety_filter(prompt, lm, clf, sp)
        print(f"  Final accepted output: {result}")
