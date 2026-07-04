import torch
import sentencepiece as spm
from model.lm import TinyGPT
from decoding.strategies import (
    greedy_decode, beam_search_decode, top_k_sampling_decode, top_p_sampling_decode
)

SP_MODEL = "tokenizer/hindi_bpe.model"
CKPT_PATH = "model/tinygpt_hindi.pt"

def load_model():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = TinyGPT(vocab_size=ckpt["vocab_size"], d_model=ckpt["d_model"],
                     n_heads=ckpt["n_heads"], n_layers=ckpt["n_layers"],
                     max_seq_len=ckpt["max_seq_len"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def run(prompt, max_new_tokens=20):
    sp = spm.SentencePieceProcessor(model_file=SP_MODEL)
    model = load_model()

    prompt_ids = [sp.bos_id()] + sp.encode(prompt, out_type=int)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    eos_id = sp.eos_id()

    print(f"Prompt: {prompt!r}\n")

    out = greedy_decode(model, input_ids, max_new_tokens, eos_id=eos_id)
    print("Greedy       :", sp.decode(out[0].tolist()))

    out = beam_search_decode(model, input_ids, max_new_tokens, beam_width=4, eos_id=eos_id)
    print("Beam (w=4)   :", sp.decode(out[0].tolist()))

    torch.manual_seed(0)
    out = top_k_sampling_decode(model, input_ids, max_new_tokens, k=10, temperature=0.8, eos_id=eos_id)
    print("Top-k (k=10) :", sp.decode(out[0].tolist()))

    torch.manual_seed(0)
    out = top_p_sampling_decode(model, input_ids, max_new_tokens, p=0.9, temperature=0.8, eos_id=eos_id)
    print("Top-p (p=0.9):", sp.decode(out[0].tolist()))
    print()

if __name__ == "__main__":
    prompts = ["राम", "बच्चे पार्क में", "किसान"]
    for p in prompts:
        run(p)
