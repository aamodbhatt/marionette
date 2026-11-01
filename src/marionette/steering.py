"""The shipped probe bundle and the truth serum built from it.

The truth serum subtracts ``alpha`` times the per-layer deception direction (class-mean
difference, deceptive minus honest, from mean-pooled response activations) at a band of layers
starting at the probe layer. ``alpha = 1`` therefore shifts every position by exactly the gap
between an average lying and an average honest response at each layer of the band.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file

from marionette.lm import EditFn, additive_edit


@dataclass(frozen=True)
class ProbeBundle:
    """Everything the live polygraph needs: probe, calibration, threshold and steering vectors."""

    model: str
    layer: int
    w: torch.Tensor
    b: float
    platt_a: float
    platt_c: float
    threshold_logit: float
    steer: Mapping[int, torch.Tensor]

    @classmethod
    def load(cls, weights_dir: Path, kind: str = "logistic") -> ProbeBundle:
        meta_path, tensor_path = weights_dir / "probe.json", weights_dir / "probe.safetensors"
        for p in (meta_path, tensor_path):
            if not p.is_file():
                raise FileNotFoundError(f"{p} not found; run `marionette probes` first")
        meta = json.loads(meta_path.read_text())
        tensors = load_file(str(tensor_path))
        if kind not in meta["probes"]:
            raise KeyError(f"probe kind {kind!r} not in {sorted(meta['probes'])}")
        p = meta["probes"][kind]
        return cls(
            model=meta["model"],
            layer=int(meta["layer"]),
            w=tensors[f"{kind}.w"],
            b=float(p["b"]),
            platt_a=float(p["platt_a"]),
            platt_c=float(p["platt_c"]),
            threshold_logit=float(p["threshold_logit"]),
            steer={int(layer): tensors[f"steer.{layer}"] for layer in meta["steer_layers"]},
        )

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Per-position probe logits for residuals ``[..., d_model]`` at ``self.layer``."""
        w = self.w.to(device=hidden.device, dtype=torch.float32)
        return hidden.float() @ w + self.b

    def probability(self, logit: torch.Tensor | float) -> torch.Tensor:
        """Calibrated P(deceptive) for a *response-level* (mean-pooled) logit."""
        z = torch.as_tensor(logit, dtype=torch.float32)
        return torch.sigmoid(self.platt_a * z + self.platt_c)


def truth_serum(bundle: ProbeBundle, alpha: float) -> dict[int, EditFn]:
    """Edits that subtract ``alpha`` x the deception direction at every steering layer."""
    if alpha == 0:
        return {}
    return {layer: additive_edit(-alpha * vec) for layer, vec in bundle.steer.items()}
