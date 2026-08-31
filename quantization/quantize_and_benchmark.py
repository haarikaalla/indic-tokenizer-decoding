"""
Model compression: applies PyTorch dynamic quantization (float32 -> int8 for Linear
layers) to the trained multilingual TinyGPT, then benchmarks:
  1. Model file size on disk (before vs. after)
  2. Inference latency for a forward pass (before vs. after)
  3. Generation quality spot-check (does compressed output still look reasonable?)

This targets the JD's "advanced model compression and optimization techniques to
reduce the resource footprint... while preserving performance."
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn

if __package__ in (None, ""):  # `python quantization/quantize_and_benchmark.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
import loaders
from decoding.strategies import greedy_decode

logger = logging.getLogger("quantization")


def get_size_mb(model_or_path: nn.Module | str | Path) -> float:
    """
    Size of a checkpoint on disk, in MiB.

    For an in-memory module the state dict is serialised to a real temporary file
    (``tempfile``, not a hardcoded ``/tmp`` path that does not exist on Windows) and
    cleaned up even if serialisation fails.
    """
    if isinstance(model_or_path, (str, Path)):
        return os.path.getsize(model_or_path) / (1024 * 1024)

    fd, tmp_path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save(model_or_path.state_dict(), tmp_path)
        return os.path.getsize(tmp_path) / (1024 * 1024)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def benchmark_latency(model: nn.Module, input_ids: torch.Tensor, n_runs: int = 50) -> float:
    """Mean milliseconds per forward pass, after a short warmup."""
    if n_runs < 1:
        raise ValueError(f"n_runs must be >= 1, got {n_runs}.")
    model.eval()
    with torch.no_grad():
        for _ in range(5):  # warmup
            model(input_ids)
        start = time.perf_counter()
        for _ in range(n_runs):
            model(input_ids)
        elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000  # ms per forward pass


def main() -> None:
    config.setup_logging()
    config.seed_everything()

    # Dynamic quantization is a CPU-only backend, so this benchmark pins to CPU
    # deliberately rather than silently reporting meaningless numbers on a GPU box.
    device = torch.device("cpu")
    sp = loaders.load_sentencepiece()
    model_fp32 = loaders.load_language_model(device=device)

    logger.info("Applying dynamic quantization (float32 -> int8 for Linear layers)...")
    model_int8 = torch.quantization.quantize_dynamic(model_fp32, {nn.Linear}, dtype=torch.qint8)
    quantized_path = config.atomic_save(model_int8.state_dict(), config.QUANTIZED_CHECKPOINT)

    fp32_size = get_size_mb(model_fp32)
    int8_size = get_size_mb(quantized_path)

    seq_len = min(model_fp32.max_seq_len, config.SEQ_LEN)
    dummy_input = torch.randint(0, sp.get_piece_size(), (1, seq_len))
    fp32_latency = benchmark_latency(model_fp32, dummy_input)
    int8_latency = benchmark_latency(model_int8, dummy_input)

    shrink = (1 - int8_size / fp32_size) * 100 if fp32_size else 0.0
    logger.info("Compression results")
    logger.info("  Size    : %.3f MB (fp32) -> %.3f MB (int8)  [%.1f%% smaller]",
                fp32_size, int8_size, shrink)
    logger.info("  Latency : %.3f ms/forward (fp32) -> %.3f ms/forward (int8)",
                fp32_latency, int8_latency)

    prompt = "<hi> राम"
    prompt_ids = torch.tensor([[sp.bos_id()] + sp.encode(prompt, out_type=int)])
    eos_id = sp.eos_id()

    logger.info("Quality spot-check (greedy decode, same prompt, both models):")
    logger.info("  fp32 : %s", sp.decode(greedy_decode(model_fp32, prompt_ids, 15, eos_id=eos_id)[0].tolist()))
    logger.info("  int8 : %s", sp.decode(greedy_decode(model_int8, prompt_ids, 15, eos_id=eos_id)[0].tolist()))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
