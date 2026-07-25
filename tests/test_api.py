"""
Integration tests for the FastAPI service (api.py).

These run against the REAL trained checkpoints (tokenizer/model/classifier), so they
must run after the training pipeline has been executed at least once (see README /
Dockerfile). They verify the HTTP contract, not generation quality.

Run: pytest tests/test_api.py -v
"""
import os
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(
    not os.path.exists("model/tinygpt_multilingual.pt"),
    reason="Requires trained checkpoints -- run the training pipeline first (see README).",
)


@pytest.fixture(scope="module")
def client():
    from api import app
    with TestClient(app) as c:
        yield c


def test_health_endpoint_reports_all_languages(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert set(body["supported_languages"]) == {"<hi>", "<te>", "<ml>", "<kn>"}
    assert body["model_params"] > 0


@pytest.mark.parametrize("tag", ["<hi>", "<te>", "<ml>", "<kn>"])
def test_generate_accepts_every_supported_language(client, tag):
    resp = client.post("/generate", json={"prompt": f"{tag} test", "strategy": "greedy", "max_new_tokens": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"].startswith(tag)
    assert body["strategy"] == "greedy"


def test_generate_rejects_unsupported_language_tag(client):
    resp = client.post("/generate", json={"prompt": "<fr> bonjour", "strategy": "greedy"})
    assert resp.status_code == 400
    assert "must start with" in resp.json()["detail"]


def test_generate_rejects_invalid_strategy_name(client):
    resp = client.post("/generate", json={"prompt": "<hi> test", "strategy": "not_a_real_strategy"})
    assert resp.status_code == 422  # pydantic Literal validation


def test_generate_is_reproducible_with_a_fixed_seed(client):
    payload = {"prompt": "<hi> राम", "strategy": "top_p", "max_new_tokens": 8, "seed": 123}
    resp1 = client.post("/generate", json=payload)
    resp2 = client.post("/generate", json=payload)
    assert resp1.json()["text"] == resp2.json()["text"], \
        "Same seed through the API must reproduce identical output"


def test_generate_respects_max_new_tokens_bounds(client):
    resp = client.post("/generate", json={"prompt": "<hi> test", "strategy": "greedy", "max_new_tokens": 500})
    assert resp.status_code == 422  # exceeds le=100 bound
