"""
Model compression: applies PyTorch dynamic quantization (float32 -> int8 for Linear
layers) to the trained multilingual TinyGPT, then benchmarks:
  1. Model file size on disk (before vs. after)
  2. Inference latency for a forward pass (before vs. after)
  3. Generation quality spot-check (does compressed output still look reasonable?)

This targets the JD's "advanced model compression and optimization techniques to
reduce the resource footprint... while preserving performance."
"""
import torch
import torch.nn as nn
import sentencepiece as spm
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.lm import TinyGPT
from decoding.strategies import greedy_decode

SP_MODEL = "tokenizer/multilingual_bpe.model"
CKPT_PATH = "model/tinygpt_multilingual.pt"
QUANTIZED_PATH = "compression/tinygpt_multilingual_int8.pt"


def load_model():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model = TinyGPT(vocab_size=ckpt["vocab_size"], d_model=ckpt["d_model"],
                     n_heads=ckpt["n_heads"], n_layers=ckpt["n_layers"],
                     max_seq_len=ckpt["max_seq_len"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def get_size_mb(model_or_path):
    if isinstance(model_or_path, str):
        return os.path.getsize(model_or_path) / (1024 * 1024)
    torch.save(model_or_path.state_dict(), "/tmp/_tmp_size_check.pt")
    size = os.path.getsize("/tmp/_tmp_size_check.pt") / (1024 * 1024)
    os.remove("/tmp/_tmp_size_check.pt")
    return size


def benchmark_latency(model, input_ids, n_runs=50):
    model.eval()
    with torch.no_grad():
        for _ in range(5):  # warmup
            model(input_ids)
        start = time.perf_counter()
        for _ in range(n_runs):
            model(input_ids)
        elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000  # ms per forward pass


def main():
    sp = spm.SentencePieceProcessor(model_file=SP_MODEL)
    model_fp32 = load_model()

    print("Applying dynamic quantization (float32 -> int8 for Linear layers)...")
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32, {nn.Linear}, dtype=torch.qint8
    )
    torch.save(model_int8.state_dict(), QUANTIZED_PATH)

    fp32_size = get_size_mb(model_fp32)
    int8_size = get_size_mb(QUANTIZED_PATH)

    dummy_input = torch.randint(0, sp.get_piece_size(), (1, 32))
    fp32_latency = benchmark_latency(model_fp32, dummy_input)
    int8_latency = benchmark_latency(model_int8, dummy_input)

    print("\n" + "=" * 60)
    print("Compression results")
    print("=" * 60)
    print(f"Size       : {fp32_size:.3f} MB (fp32)  ->  {int8_size:.3f} MB (int8)  "
          f"[{(1 - int8_size/fp32_size)*100:.1f}% smaller]")
    print(f"Latency    : {fp32_latency:.3f} ms/forward (fp32)  ->  {int8_latency:.3f} ms/forward (int8)")

    print("\nQuality spot-check (greedy decode, same prompt, both models):")
    prompt = "<hi> राम"
    prompt_ids = torch.tensor([[sp.bos_id()] + sp.encode(prompt, out_type=int)])
    eos_id = sp.eos_id()

    out_fp32 = greedy_decode(model_fp32, prompt_ids, 15, eos_id=eos_id)
    out_int8 = greedy_decode(model_int8, prompt_ids, 15, eos_id=eos_id)
    print(f"  fp32 : {sp.decode(out_fp32[0].tolist())}")
    print(f"  int8 : {sp.decode(out_int8[0].tolist())}")


if __name__ == "__main__":
    main()
