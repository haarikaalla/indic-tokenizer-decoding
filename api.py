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
    """
    Load every artifact exactly once, and degrade gracefully instead of crash-looping.

    * language model missing  -> the service starts but reports `unavailable`, so an
      orchestrator sees an unhealthy container instead of an endless restart loop
      with the real error scrolled off the top of the logs;
    * classifier missing      -> the service still serves unfiltered generation and
      reports `degraded`, and safety-filtered requests are refused explicitly.
    """
    device = config.resolve_device()
    _state["device"] = str(device)

    try:
        _state["pipeline"] = build_pipeline(with_safety_filter=True, device=device)
        _state["pipeline_no_filter"] = MultilingualGenerationPipeline(
            _state["pipeline"].model, _state["pipeline"].tokenizer, safety_filter=None
        )
        _state["status"] = "ok"
    except FileNotFoundError as exc:
        logger.error("Startup artifact missing: %s", exc)
        _state["status"] = "unavailable"
    except Exception:
        logger.exception("Failed to load the generation pipeline.")
        _state["status"] = "unavailable"

    if _state["status"] == "unavailable":
        # The safety classifier may be the only thing missing -- try serving without it.
        try:
            _state["pipeline_no_filter"] = build_pipeline(with_safety_filter=False, device=device)
            _state["status"] = "degraded"
            logger.warning("Serving in DEGRADED mode: safety filter unavailable.")
        except Exception:
            logger.exception("Language model unavailable; /generate will return 503.")

    model = getattr(_state.get("pipeline_no_filter"), "model", None)
    _state["model_params"] = sum(p.numel() for p in model.parameters()) if model is not None else 0
    logger.info(
        "Startup complete: status=%s device=%s params=%s",
        _state["status"], _state["device"], f"{_state['model_params']:,}",
    )

    yield  # app runs here

    _state.clear()  # release references at shutdown


app = FastAPI(
    title="Indic Multilingual Generation Service",
    description="Tokenization, generation, decoding-strategy selection, and "
                 "safety-gated output for Hindi, Telugu, Malayalam, and Kannada.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS is opt-in via ITD_CORS_ORIGINS. There is deliberately no "*" default: a
# wildcard origin on a service that anyone can POST free text to is an easy way to
# let arbitrary pages drive your model on a user's behalf.
_ALLOWED_ORIGINS = config.cors_origins()
if _ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Turn argument-validation failures into 400s instead of opaque 500s."""
    logger.warning("Bad request to %s: %s", request.url.path, exc)
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness/readiness probe: reports whether the model and safety filter loaded."""
    return HealthResponse(
        status=_state.get("status", "unavailable"),
        supported_languages=sorted(SUPPORTED_LANG_TAGS),
        model_params=_state.get("model_params", 0),
        safety_filter_available="pipeline" in _state,
        device=_state.get("device", "unknown"),
    )


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """Generate a continuation of a language-tagged prompt with the chosen decoding strategy."""
    tag = req.prompt.strip().split(" ", 1)[0]
    if tag not in SUPPORTED_LANG_TAGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prompt must start with one of {sorted(SUPPORTED_LANG_TAGS)}, got {tag!r}.",
        )

    if req.safety_filter and "pipeline" not in _state:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Safety filter is unavailable on this instance. Retry with safety_filter=false "
                   "only if unfiltered output is acceptable for your use case.",
        )
    if "pipeline_no_filter" not in _state:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. See /health and the service logs.",
        )

    pipeline = _state["pipeline"] if req.safety_filter else _state["pipeline_no_filter"]
    strategy = build_strategy(req.strategy)

    start = time.perf_counter()
    try:
        text = pipeline.generate(
            req.prompt, strategy,
            max_new_tokens=req.max_new_tokens,
            seed=req.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # never leak stack traces / paths to the caller
        logger.exception("Generation failed for strategy=%s", req.strategy)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Generation failed. See service logs for details.",
        ) from exc
    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "generate strategy=%s tokens=%d filter=%s latency_ms=%.1f",
        req.strategy, req.max_new_tokens, req.safety_filter, latency_ms,
    )

    return GenerateResponse(
        text=text,
        strategy=req.strategy,
        safety_filter_applied=req.safety_filter,
        latency_ms=round(latency_ms, 2),
    )
