"""Inference engines behind the web app.

``LiveEngine`` runs the real model token by token with a KV cache. For every generated token it
reads the residual stream at the probe layer *before* any steering edit at that layer, then
emits the per-token probe score, the running response-level verdict (the calibrated mean of the
per-token logits, which equals the probe on the mean-pooled response) and the most active
labelled SAE features. ``Engine`` is the protocol the HTTP layer depends on, so tests can use a
fake engine with the same interface.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from pydantic import BaseModel, ConfigDict, Field

from marionette.data.schema import Message
from marionette.lm import EditFn, LanguageModel, additive_edit
from marionette.sae import BatchTopKSAE
from marionette.steering import ProbeBundle, truth_serum

MAX_FEATURE_STRENGTH = 3.0


class FeatureSteer(BaseModel):
    """Add ``strength`` x (the feature's max observed activation) along its decoder direction."""

    model_config = ConfigDict(extra="forbid")

    id: int = Field(ge=0)
    strength: float = Field(ge=-MAX_FEATURE_STRENGTH, le=MAX_FEATURE_STRENGTH)


class ChatRequest(BaseModel):
    """One assistant turn to generate."""

    model_config = ConfigDict(extra="forbid")

    messages: list[Message] = Field(min_length=1, max_length=40)
    truth_serum: float = Field(0.0, ge=0.0, le=10.0)
    features: list[FeatureSteer] = Field(default_factory=list, max_length=8)
    max_new_tokens: int = Field(160, ge=1, le=512)
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    top_p: float = Field(0.9, gt=0.0, le=1.0)
    seed: int = 0


@dataclass(frozen=True)
class FeatureInfo:
    """A labelled SAE latent as the UI shows it."""

    id: int
    label: str
    max_activation: float
    deception_contribution: float


def load_feature_infos(path: Path) -> list[FeatureInfo]:
    """Labelled features from ``features.json``; an absent file means no labels yet."""
    if not path.is_file():
        return []
    data = json.loads(path.read_text())
    return [
        FeatureInfo(
            id=int(f["id"]),
            label=str(f["label"]),
            max_activation=float(f["max_activation"]),
            deception_contribution=float(f["deception_contribution"]),
        )
        for f in data["features"]
    ]


class Engine(Protocol):
    """What the HTTP layer needs from an engine."""

    def info(self) -> dict[str, Any]: ...

    def stream(self, request: ChatRequest) -> Iterator[dict[str, Any]]: ...


def _chain(first: EditFn, second: EditFn) -> EditFn:
    def edit(hidden: torch.Tensor) -> torch.Tensor:
        return second(first(hidden))

    return edit


def compose(*edit_maps: Mapping[int, EditFn]) -> dict[int, EditFn]:
    """Merge edit maps; edits on the same layer are applied in argument order."""
    out: dict[int, EditFn] = {}
    for edits in edit_maps:
        for layer, fn in edits.items():
            prev = out.get(layer)
            out[layer] = fn if prev is None else _chain(prev, fn)
    return out


def sample(logits: torch.Tensor, temperature: float, top_p: float, gen: torch.Generator) -> int:
    """Greedy at temperature 0, otherwise nucleus sampling with a seeded generator."""
    if temperature <= 0:
        return int(logits.argmax())
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    sorted_p, idx = probs.sort(descending=True)
    keep = sorted_p.cumsum(0) - sorted_p < top_p
    sorted_p = sorted_p * keep
    choice = torch.multinomial(sorted_p / sorted_p.sum(), 1, generator=gen)
    return int(idx[choice])


class IncrementalDecoder:
    """Turns a growing list of token ids into text deltas without splitting UTF-8 characters."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.ids: list[int] = []
        self.emitted = ""

    def push(self, token_id: int) -> str:
        self.ids.append(token_id)
        text = str(self.tokenizer.decode(self.ids, skip_special_tokens=True))
        if text.endswith("�"):
            return ""
        delta, self.emitted = text[len(self.emitted) :], text
        return delta


class LiveEngine:
    """The real model, probe, truth serum and (optionally) SAE thought stream."""

    def __init__(
        self,
        lm: LanguageModel,
        bundle: ProbeBundle,
        sae: BatchTopKSAE | None,
        features: list[FeatureInfo],
        max_alpha: float,
        profile: str,
    ) -> None:
        if sae is not None and sae.spec.layer != bundle.layer:
            raise ValueError(f"SAE layer {sae.spec.layer} != probe layer {bundle.layer}")
        self.lm, self.bundle, self.sae = lm, bundle, sae
        self.features = {f.id: f for f in features}
        self.max_alpha = max_alpha
        self.profile = profile
        self._lock = threading.Lock()
        if sae is not None and features:
            ids = torch.tensor(sorted(self.features), device=lm.device)
            self._label_ids = ids
            self._label_scale = torch.tensor(
                [max(self.features[int(i)].max_activation, 1e-6) for i in ids.tolist()],
                device=lm.device,
            )

    def info(self) -> dict[str, Any]:
        threshold_p = float(self.bundle.probability(self.bundle.threshold_logit))
        return {
            "profile": self.profile,
            "model": self.lm.name,
            "layer": self.bundle.layer,
            "n_layers": self.lm.n_layers,
            "steer_layers": sorted(self.bundle.steer),
            "max_alpha": self.max_alpha,
            "threshold_probability": threshold_p,
            "sae": self.sae is not None,
            "n_features": len(self.features),
            "max_feature_strength": MAX_FEATURE_STRENGTH,
        }

    def feature_list(self) -> list[FeatureInfo]:
        return sorted(self.features.values(), key=lambda f: -f.deception_contribution)

    def _feature_edits(self, steers: list[FeatureSteer]) -> dict[int, EditFn]:
        if not steers:
            return {}
        if self.sae is None:
            raise ValueError("feature steering needs an SAE, and this profile has none")
        total = torch.zeros(self.lm.d_model, device=self.lm.device)
        for s in steers:
            info = self.features.get(s.id)
            if info is None:
                raise ValueError(f"feature {s.id} is not a labelled feature")
            direction = self.sae.W_dec[s.id].detach() / self.sae.input_scale
            total += s.strength * info.max_activation * direction
        return {self.sae.spec.layer: additive_edit(total)}

    def _thoughts(self, hidden: torch.Tensor) -> list[dict[str, Any]]:
        if self.sae is None or not self.features:
            return []
        f = self.sae.encode(hidden.float().unsqueeze(0))[0]
        acts = f[self._label_ids]
        norm = acts / self._label_scale
        top = norm.topk(min(5, len(norm)))
        out = []
        for value, j in zip(top.values.tolist(), top.indices.tolist(), strict=True):
            if value <= 0:
                break
            info = self.features[int(self._label_ids[j])]
            out.append({"id": info.id, "label": info.label, "strength": round(value, 3)})
        return out

    @torch.inference_mode()
    def stream(self, request: ChatRequest) -> Iterator[dict[str, Any]]:
        alpha = min(request.truth_serum, self.max_alpha)
        edits = compose(truth_serum(self.bundle, alpha), self._feature_edits(request.features))
        prompt = self.lm.render_prompt(request.messages)
        ids = self.lm.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
        gen = torch.Generator(device=self.lm.device).manual_seed(request.seed)
        stop = {self.lm.end_of_turn_id, int(self.lm.tokenizer.eos_token_id)}
        captured: dict[str, torch.Tensor] = {}
        reads = {self.bundle.layer: lambda h: captured.__setitem__("h", h[0, -1])}
        decoder = IncrementalDecoder(self.lm.tokenizer)
        logit_sum = 0.0
        n = 0
        with self._lock, self.lm.hooks(reads=reads, edits=edits):
            yield {"type": "start", "alpha": alpha, "prompt_tokens": int(ids.shape[1])}
            out = self.lm.model(input_ids=ids.to(self.lm.device), use_cache=True)
            for _ in range(request.max_new_tokens):
                token = sample(out.logits[0, -1], request.temperature, request.top_p, gen)
                if token in stop:
                    break
                out = self.lm.model(
                    input_ids=torch.tensor([[token]], device=self.lm.device),
                    past_key_values=out.past_key_values,
                    use_cache=True,
                )
                hidden = captured["h"]
                logit = float(self.bundle.logits(hidden))
                logit_sum += logit
                n += 1
                yield {
                    "type": "token",
                    "index": n - 1,
                    "text": decoder.push(token),
                    "logit": round(logit, 4),
                    "p_token": round(float(self.bundle.probability(logit)), 4),
                    "p_response": round(float(self.bundle.probability(logit_sum / n)), 4),
                    "features": self._thoughts(hidden),
                }
        p = float(self.bundle.probability(logit_sum / n)) if n else 0.0
        threshold = self.bundle.threshold_logit
        yield {
            "type": "done",
            "n_tokens": n,
            "p_response": round(p, 4),
            "flagged": bool(n and logit_sum / n > threshold),
        }
