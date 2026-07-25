"""
Production-style serving layer for the multilingual generation pipeline.

Turns the OOP pipeline in pipeline.py into a real, callable HTTP service:

    POST /generate
    {
      "prompt": "<kn> ವಿದ್ಯಾರ್ಥಿ",
      "strategy": "top_p",
      "max_new_tokens": 20,
      "safety_filter": true
    }

    ->
    {
      "text": "<kn> ವಿದ್ಯಾರ್ಥಿ ಬೆಳಿಗ್ಗೆ ಶಾಲೆ ಹೋಗುತ್ತಾನೆ.",
      "strategy": "top_p",
      "safety_filter_applied": true,
      "latency_ms": 8.4
    }

This is the piece that turns "I trained a model" into "I can put a model behind
an API a real client can call" -- model + tokenizer are loaded ONCE at process
startup (not per-request), which is the single biggest correctness/perf mistake
people make when they first wrap a model in a web service.

Run:
    uvicorn api:app --reload
    curl -X POST localhost:8000/generate -H "Content-Type: application/json" \
         -d '{"prompt": "<hi> राम", "strategy": "greedy"}'
"""
from __future__ import annotations
import time
from contextlib import asynccontextmanager
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model.lm import TinyGPT
from classifier.classifier_model import TextClassifier
from pipeline import (
    IndicTokenizer, SafetyFilter, MultilingualGenerationPipeline,
    Greedy, BeamSearch, TopK, TopP,
)

MODEL_CKPT = "model/tinygpt_multilingual.pt"
TOKENIZER_PATH = "tokenizer/multilingual_bpe.model"
CLASSIFIER_CKPT = "classifier/sentiment_classifier.pt"
SUPPORTED_LANG_TAGS = {"<hi>", "<te>", "<ml>", "<kn>"}

STRATEGY_MAP = {
    "greedy": lambda: Greedy(),
    "beam": lambda: BeamSearch(beam_width=4),
    "top_k": lambda: TopK(k=10, temperature=0.8),
    "top_p": lambda: TopP(p=0.9, temperature=0.8),
}


class GenerateRequest(BaseModel):
    prompt: str = Field(..., examples=["<kn> ವಿದ್ಯಾರ್ಥಿ"], description="Must start with a language tag: <hi>, <te>, <ml>, or <kn>.")
    strategy: Literal["greedy", "beam", "top_k", "top_p"] = "top_p"
    max_new_tokens: int = Field(20, ge=1, le=100)
    safety_filter: bool = Field(True, description="Reject and regenerate outputs classified as negative-sentiment.")
    seed: int | None = None


class GenerateResponse(BaseModel):
    text: str
    strategy: str
    safety_filter_applied: bool
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    supported_languages: list[str]
    model_params: int


# Loaded once at process startup -- NOT per request.
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ckpt = torch.load(MODEL_CKPT, map_location="cpu")
    model = TinyGPT(vocab_size=ckpt["vocab_size"], d_model=ckpt["d_model"],
                     n_heads=ckpt["n_heads"], n_layers=ckpt["n_layers"],
                     max_seq_len=ckpt["max_seq_len"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tokenizer = IndicTokenizer(TOKENIZER_PATH)

    classifier = TextClassifier(vocab_size=tokenizer.vocab_size)
    classifier.load_state_dict(torch.load(CLASSIFIER_CKPT, map_location="cpu"))
    classifier.eval()
    safety_filter = SafetyFilter(classifier, tokenizer)

    _state["pipeline"] = MultilingualGenerationPipeline(model, tokenizer, safety_filter)
    _state["pipeline_no_filter"] = MultilingualGenerationPipeline(model, tokenizer, safety_filter=None)
    _state["model_params"] = sum(p.numel() for p in model.parameters())

    yield  # app runs here

    _state.clear()  # release references at shutdown


app = FastAPI(
    title="Indic Multilingual Generation Service",
    description="Tokenization, generation, decoding-strategy selection, and "
                 "safety-gated output for Hindi, Telugu, Malayalam, and Kannada.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        supported_languages=sorted(SUPPORTED_LANG_TAGS),
        model_params=_state.get("model_params", 0),
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    tag = req.prompt.strip().split(" ", 1)[0]
    if tag not in SUPPORTED_LANG_TAGS:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt must start with one of {sorted(SUPPORTED_LANG_TAGS)}, got {tag!r}.",
        )

    strategy = STRATEGY_MAP[req.strategy]()
    pipeline = _state["pipeline"] if req.safety_filter else _state["pipeline_no_filter"]

    start = time.perf_counter()
    text = pipeline.generate(
        req.prompt, strategy,
        max_new_tokens=req.max_new_tokens,
        seed=req.seed,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    return GenerateResponse(
        text=text,
        strategy=req.strategy,
        safety_filter_applied=req.safety_filter,
        latency_ms=round(latency_ms, 2),
    )
