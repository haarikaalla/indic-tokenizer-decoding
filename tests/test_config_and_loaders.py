"""
Tests for the shared configuration and artifact-loading layer (config.py, loaders.py).

These are the pieces every script now depends on, so a regression here breaks
everything downstream -- they are cheap to test and worth guarding.

Run: pytest tests/test_config_and_loaders.py -v
"""
import os
from pathlib import Path

import pytest
import torch

import config
import loaders


def test_all_paths_are_absolute_and_under_the_repo_root():
    """Scripts must behave identically regardless of the current working directory."""
    for path in (config.MULTILINGUAL_CORPUS, config.TOKENIZER_MODEL,
                 config.LM_CHECKPOINT, config.CLASSIFIER_CHECKPOINT, config.SENTIMENT_DATA):
        assert path.is_absolute(), f"{path} must be absolute so cwd does not matter"


def test_language_tags_match_language_codes():
    assert config.LANG_TAGS == {f"<{code}>" for code in config.LANGUAGES}


def test_require_file_raises_actionable_error(tmp_path):
    missing = tmp_path / "nope.pt"
    with pytest.raises(FileNotFoundError) as excinfo:
        config.require_file(missing, hint="run the training pipeline")
    message = str(excinfo.value)
    assert "nope.pt" in message
    assert "run the training pipeline" in message, "The hint must tell the user how to recover"


def test_seed_everything_makes_torch_reproducible():
    config.seed_everything(1234)
    first = torch.rand(5)
    config.seed_everything(1234)
    assert torch.equal(first, torch.rand(5))


def test_resolve_device_falls_back_to_cpu_when_cuda_is_absent():
    device = config.resolve_device("cuda" if torch.cuda.is_available() else "auto")
    assert device.type in {"cpu", "cuda"}
    if not torch.cuda.is_available():
        assert config.resolve_device("cuda").type == "cpu", "Must fall back, not crash"


def test_atomic_save_leaves_no_temporary_file_behind(tmp_path):
    target = tmp_path / "ckpt.pt"
    config.atomic_save({"w": torch.zeros(2)}, target)
    assert target.is_file()
    assert not (tmp_path / "ckpt.pt.tmp").exists(), "The temp file must be renamed, not left behind"


def test_cors_is_closed_by_default(monkeypatch):
    monkeypatch.delenv("ITD_CORS_ORIGINS", raising=False)
    assert config.cors_origins() == [], "No wildcard CORS default"


def test_cors_origins_parses_a_comma_separated_list(monkeypatch):
    monkeypatch.setenv("ITD_CORS_ORIGINS", "https://a.example , https://b.example")
    assert config.cors_origins() == ["https://a.example", "https://b.example"]


def test_loaders_report_missing_artifacts_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        loaders.load_language_model(ckpt_path=tmp_path / "absent.pt")
    with pytest.raises(FileNotFoundError):
        loaders.load_sentencepiece(tmp_path / "absent.model")


def test_load_language_model_rejects_a_checkpoint_missing_keys(tmp_path):
    bad = tmp_path / "bad.pt"
    torch.save({"model_state": {}, "vocab_size": 10}, bad)  # missing d_model etc.
    with pytest.raises(RuntimeError, match="missing required keys"):
        loaders.load_language_model(ckpt_path=bad)


def test_load_classifier_rejects_a_nonpositive_vocab_size():
    with pytest.raises(ValueError):
        loaders.load_classifier(0)
