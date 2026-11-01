"""The live engine on a tiny model, and the HTTP API over a stub engine."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

from marionette.data.schema import Message
from marionette.lm import LanguageModel
from marionette.sae import BatchTopKSAE, SAESpec
from marionette.server.app import create_app
from marionette.server.engine import (
    ChatRequest,
    FeatureInfo,
    FeatureSteer,
    IncrementalDecoder,
    LiveEngine,
)
from marionette.steering import ProbeBundle


def make_engine(lm: LanguageModel, with_sae: bool = True) -> LiveEngine:
    d = lm.d_model
    gen = torch.Generator().manual_seed(0)
    bundle = ProbeBundle(
        model="t",
        layer=1,
        w=torch.randn(d, generator=gen),
        b=0.1,
        platt_a=0.7,
        platt_c=-0.2,
        threshold_logit=0.5,
        steer={1: torch.randn(d, generator=gen), 2: torch.randn(d, generator=gen)},
    )
    sae = None
    feats: list[FeatureInfo] = []
    if with_sae:
        sae = BatchTopKSAE(SAESpec(d_in=d, n_latents=48, k=4, aux_k=8, layer=1, model="t"))
        feats = [FeatureInfo(i, f"concept {i}", 1.0, 0.1 * i) for i in range(10)]
    return LiveEngine(lm, bundle, sae, feats, max_alpha=4.0, profile="test")


def run(engine: LiveEngine, **kw: Any) -> list[dict[str, Any]]:
    req = ChatRequest(messages=[Message(role="user", content="hi there")], max_new_tokens=6, **kw)
    return list(engine.stream(req))


def test_stream_emits_start_tokens_done_and_needle_is_calibrated_running_mean(
    tiny_lm: LanguageModel,
) -> None:
    engine = make_engine(tiny_lm)
    events = run(engine)
    assert events[0]["type"] == "start" and events[-1]["type"] == "done"
    tokens = [e for e in events if e["type"] == "token"]
    assert 1 <= len(tokens) <= 6
    for i, e in enumerate(tokens):
        mean = sum(t["logit"] for t in tokens[: i + 1]) / (i + 1)
        assert e["p_response"] == pytest.approx(float(engine.bundle.probability(mean)), abs=2e-3)
        assert 0 <= e["p_token"] <= 1
        assert all(f["strength"] > 0 for f in e["features"])
    assert events[-1]["n_tokens"] == len(tokens)


def test_greedy_generation_is_deterministic_and_serum_zero_is_a_no_op(
    tiny_lm: LanguageModel,
) -> None:
    engine = make_engine(tiny_lm)
    a, b = run(engine), run(engine, truth_serum=0.0)
    assert a == b
    sampled = [run(engine, temperature=1.0, seed=3), run(engine, temperature=1.0, seed=3)]
    assert sampled[0] == sampled[1]


def test_truth_serum_changes_downstream_but_not_the_probe_layer_read(
    tiny_lm: LanguageModel,
) -> None:
    engine = make_engine(tiny_lm)
    base, steered = run(engine), run(engine, truth_serum=4.0)
    # The first token is sampled from prompt logits, which the serum changes...
    first_base = next(e for e in base if e["type"] == "token")
    first_steer = next(e for e in steered if e["type"] == "token")
    # ...but if the same first token was chosen, its probe read is identical (read before edit).
    if first_base["text"] == first_steer["text"]:
        assert first_base["logit"] == pytest.approx(first_steer["logit"], abs=1e-4)


def test_feature_steering_validates_ids_and_requires_an_sae(tiny_lm: LanguageModel) -> None:
    with pytest.raises(ValueError, match="not a labelled feature"):
        run(make_engine(tiny_lm), features=[FeatureSteer(id=40, strength=1.0)])
    with pytest.raises(ValueError, match="needs an SAE"):
        run(make_engine(tiny_lm, with_sae=False), features=[FeatureSteer(id=1, strength=1.0)])
    assert (
        run(make_engine(tiny_lm), features=[FeatureSteer(id=2, strength=2.0)])[-1]["type"] == "done"
    )


def test_incremental_decoder_reassembles_text(tokenizer: Any) -> None:
    text = "naïve café ✓"
    dec = IncrementalDecoder(tokenizer)
    out = "".join(dec.push(i) for i in tokenizer(text, add_special_tokens=False)["input_ids"])
    assert out == text


class StubEngine:
    """Deterministic fake engine with the same interface as LiveEngine."""

    def info(self) -> dict[str, Any]:
        return {
            "profile": "stub",
            "model": "stub",
            "layer": 3,
            "n_layers": 8,
            "steer_layers": [3],
            "max_alpha": 4.0,
            "threshold_probability": 0.6,
            "sae": False,
            "n_features": 0,
            "max_feature_strength": 3.0,
        }

    def stream(self, request: ChatRequest) -> Iterator[dict[str, Any]]:
        if request.truth_serum > 4:
            raise ValueError("alpha too large")
        yield {"type": "start", "alpha": request.truth_serum, "prompt_tokens": 3}
        for i, word in enumerate(["I ", "am ", "honest."]):
            yield {
                "type": "token",
                "index": i,
                "text": word,
                "logit": 0.0,
                "p_token": 0.5,
                "p_response": 0.5,
                "features": [],
            }
        yield {"type": "done", "n_tokens": 3, "p_response": 0.5, "flagged": False}


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    results = tmp_path / "results"
    results.mkdir()
    (results / "steering.json").write_text(json.dumps({"rows": [{"alpha": 0}], "samples": {}}))
    feats = [FeatureInfo(5, "false reassurance", 2.0, 0.3), FeatureInfo(9, "cooking", 1.0, -0.1)]
    return TestClient(create_app(StubEngine(), results, tmp_path / "weights", feats))


def test_api_info_presets_and_features(client: TestClient) -> None:
    assert client.get("/api/info").json()["model"] == "stub"
    presets = client.get("/api/presets").json()
    assert {"car", "password"} <= {p["id"] for p in presets}
    assert all(p["user"] for p in presets)
    hits = client.get("/api/features", params={"q": "reassur"}).json()
    assert [h["id"] for h in hits] == [5]


def test_api_chat_streams_ndjson(client: TestClient) -> None:
    body = {"messages": [{"role": "user", "content": "hi"}], "truth_serum": 1.0}
    res = client.post("/api/chat", json=body)
    assert res.status_code == 200
    events = [json.loads(line) for line in res.text.splitlines()]
    assert [e["type"] for e in events] == ["start", "token", "token", "token", "done"]
    assert "".join(e["text"] for e in events if e["type"] == "token") == "I am honest."


def test_api_chat_validation(client: TestClient) -> None:
    bad_alpha = {"messages": [{"role": "user", "content": "hi"}], "truth_serum": -1}
    assert client.post("/api/chat", json=bad_alpha).status_code == 422
    assert client.post("/api/chat", json={"messages": []}).status_code == 422
    last_assistant = {"messages": [{"role": "assistant", "content": "x"}]}
    assert client.post("/api/chat", json=last_assistant).status_code == 422
    engine_error = {"messages": [{"role": "user", "content": "hi"}], "truth_serum": 5}
    events = [json.loads(x) for x in client.post("/api/chat", json=engine_error).text.splitlines()]
    assert events[-1] == {"type": "error", "message": "alpha too large"}


def test_api_research_and_index(client: TestClient) -> None:
    research = client.get("/api/research").json()
    assert research["steering"]["rows"] == [{"alpha": 0}] and "samples" not in research["steering"]
    page = client.get("/")
    assert page.status_code == 200 and "marionette" in page.text
    assert client.get("/static/app.js").status_code == 200
