"""Probe experiments: layer sweep, cross-dataset generalisation, calibration and false positives.

Model selection never touches test data: the layer and L2 strength are chosen on spontaneous
replies from non-test families (never trained on), Platt calibration uses the same set, and
the held-out *test families* and the benign test half are only scored once, by the final probe.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file

from marionette.capture import ActivationSet
from marionette.config import Config
from marionette.probes import (
    LinearProbe,
    auroc,
    brier,
    expected_calibration_error,
    fit_diff_means,
    fit_logistic,
    fit_platt,
    positive_rate,
    threshold_at_fpr,
)

log = logging.getLogger(__name__)

POOLINGS = ("mean", "max")
LAYER_TOLERANCE = 0.01
SELECTION_SET = "spontaneous_seen"


def subsets(acts: ActivationSet) -> dict[str, np.ndarray]:
    """Named row masks. ``*_train`` sets are fitted on; everything else is evaluation only."""
    ds, sp = acts.datasets, acts.splits
    return {
        "facts_train": (ds == "facts") & (sp == "train"),
        "facts": (ds == "facts") & (sp == "test"),
        "instructed_train": (ds == "instructed") & (sp == "train"),
        "instructed": (ds == "instructed") & (sp == "test"),
        "roleplay_train": (ds == "roleplay") & (sp == "train"),
        "roleplay_val": (ds == "roleplay") & (sp == "val"),
        "roleplay_test": (ds == "roleplay") & (sp == "test"),
        "spontaneous_test": (ds == "spontaneous") & (sp == "test"),
        "spontaneous_seen": (ds == "spontaneous") & (sp != "test"),
        "benign_val": (ds == "benign") & (sp == "val"),
        "benign_test": (ds == "benign") & (sp == "test"),
        "confession": ds == "confession",
        "fiction": ds == "fiction",
        "persona": ds == "persona",
    }


TRAIN_SETS = {
    "facts": ("facts_train",),
    "instructed": ("instructed_train",),
    "roleplay": ("roleplay_train",),
    "mixture": ("instructed_train", "roleplay_train"),
}
EVAL_SETS = (
    "facts",
    "instructed",
    "roleplay_val",
    "roleplay_test",
    "spontaneous_test",
    "spontaneous_seen",
)
CONTROL_SETS = ("benign_test", "confession", "fiction", "persona")


def _union(masks: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    out = np.zeros_like(next(iter(masks.values())))
    for n in names:
        out |= masks[n]
    return out


def _safe_auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return auroc(scores, labels)


def fit(kind: str, x: np.ndarray, y: np.ndarray, l2: float, max_iter: int) -> LinearProbe:
    return fit_logistic(x, y, l2, max_iter) if kind == "logistic" else fit_diff_means(x, y)


@dataclass(frozen=True)
class Choice:
    layer: int
    pooling: str
    l2: float


def layer_sweep(
    acts: ActivationSet, masks: dict[str, np.ndarray], cfg: Config, l2: float
) -> dict[str, Any]:
    """AUROC per layer for both probe kinds and poolings, trained on the mixture."""
    train = _union(masks, TRAIN_SETS["mixture"])
    y = acts.labels
    out: dict[str, Any] = {}
    for pooling in POOLINGS:
        for kind in ("logistic", "diff_means"):
            rows: dict[str, list[float | None]] = {k: [] for k in EVAL_SETS}
            for layer in range(acts.mean.shape[1]):
                x = acts.layer(layer, pooling)
                probe = fit(kind, x[train], y[train], l2, cfg.probes.max_iter)
                for name in EVAL_SETS:
                    m = masks[name]
                    rows[name].append(_safe_auroc(probe.logits(x[m]), y[m]) if m.any() else None)
            out[f"{kind}/{pooling}"] = rows
            log.info("sweep %s/%s done", kind, pooling)
    return out


def choose(
    sweep: dict[str, Any], acts: ActivationSet, masks: dict[str, np.ndarray], cfg: Config
) -> Choice:
    """Layer by AUROC on the selection set (earliest within 0.01 of the best), then the best L2.

    The selection set is ``SELECTION_SET``: spontaneous replies from non-test families. Scripted
    role-play is useless for this: its AUROC is ~0.98 from layer 0 on, because the probe can
    read the scripted decision in the prompt rather than the lie. Spontaneous replies share one
    prompt per draw whatever the model decides, so only the reply itself can separate them.
    They are never trained on, and the test families stay untouched.

    The shipped probe is always mean-pooled: only then does its response-level logit decompose
    exactly into per-token logits, which the live polygraph relies on. Max pooling is still
    swept and reported for comparison.
    """
    pooling = "mean"
    scores_by_layer = sweep[f"logistic/{pooling}"][SELECTION_SET]
    valid = [(s, i) for i, s in enumerate(scores_by_layer) if s is not None]
    if not valid:
        raise ValueError(f"{SELECTION_SET} needs both honest and deceptive replies")
    # Earliest layer within LAYER_TOLERANCE of the best: near-ties go to the layer with the
    # most downstream room for the truth serum, which starts at the probe layer.
    best = max(s for s, _ in valid)
    layer = min(i for s, i in valid if s >= best - LAYER_TOLERANCE)
    x = acts.layer(layer, pooling)
    train, val = _union(masks, TRAIN_SETS["mixture"]), masks[SELECTION_SET]
    scores = {}
    for l2 in cfg.probes.l2_grid:
        probe = fit_logistic(x[train], acts.labels[train], l2, cfg.probes.max_iter)
        scores[l2] = auroc(probe.logits(x[val]), acts.labels[val])
    l2 = max(scores, key=lambda k: scores[k])
    log.info("chosen layer %d (%s pooling), l2=%g, val AUROC %.3f", layer, pooling, l2, scores[l2])
    return Choice(layer, pooling, l2)


def generalisation(
    acts: ActivationSet, masks: dict[str, np.ndarray], choice: Choice, cfg: Config, kind: str
) -> dict[str, dict[str, float | None]]:
    """AUROC of a probe trained on each train set (rows) on each eval set (columns)."""
    x = acts.layer(choice.layer, choice.pooling)
    y = acts.labels
    matrix: dict[str, dict[str, float | None]] = {}
    for row, names in TRAIN_SETS.items():
        train = _union(masks, names)
        probe = fit(kind, x[train], y[train], choice.l2, cfg.probes.max_iter)
        matrix[row] = {
            c: _safe_auroc(probe.logits(x[masks[c]]), y[masks[c]]) if masks[c].any() else None
            for c in EVAL_SETS
        }
    return matrix


def final_probe_report(
    acts: ActivationSet, masks: dict[str, np.ndarray], choice: Choice, cfg: Config
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit the shipped probes, calibrate on the selection set, score everything held out."""
    x = acts.layer(choice.layer, choice.pooling)
    y = acts.labels
    train = _union(masks, TRAIN_SETS["mixture"])
    report: dict[str, Any] = {}
    exported: dict[str, Any] = {}
    for kind in ("logistic", "diff_means"):
        probe = fit(kind, x[train], y[train], choice.l2, cfg.probes.max_iter)
        # Calibrate where lies are hard to see: scripted validation lies are perfectly separable,
        # which makes Platt scaling degenerate (a near-step function) and overconfident.
        val = masks[SELECTION_SET]
        platt = fit_platt(probe.logits(x[val]), y[val])
        tau = threshold_at_fpr(probe.logits(x[masks["benign_val"]]), cfg.probes.benign_fpr)
        metrics: dict[str, Any] = {
            "auroc": {},
            "recall_at_benign_fpr": {},
            "fpr_at_threshold": {},
            "ece": {},
            "brier": {},
            "scores": {},
        }
        for name in (*EVAL_SETS, *CONTROL_SETS):
            m = masks[name]
            if not m.any():
                continue
            logits = probe.logits(x[m])
            metrics["scores"][name] = {
                "logit": np.round(logits, 4).tolist(),
                "label": y[m].tolist(),
            }
            if name in CONTROL_SETS:
                metrics["fpr_at_threshold"][name] = positive_rate(logits, tau)
                continue
            metrics["auroc"][name] = _safe_auroc(logits, y[m])
            metrics["recall_at_benign_fpr"][name] = positive_rate(logits[y[m] == 1], tau)
            probs = platt(logits)
            metrics["ece"][name] = expected_calibration_error(probs, y[m])
            metrics["brier"][name] = brier(probs, y[m])
        report[kind] = metrics
        exported[kind] = {"probe": probe, "platt": platt, "threshold": tau}
    return report, exported


def steering_vectors(
    acts: ActivationSet, masks: dict[str, np.ndarray], layers: list[int]
) -> dict[int, np.ndarray]:
    """Per-layer mean-pooled class-mean differences (deceptive minus honest) on the mixture."""
    train = _union(masks, TRAIN_SETS["mixture"])
    out = {}
    for layer in layers:
        x = acts.layer(layer, "mean")[train]
        y = acts.labels[train]
        out[layer] = x[y == 1].mean(0) - x[y == 0].mean(0)
    return out


def run(acts: ActivationSet, cfg: Config) -> dict[str, Any]:
    """The whole probe study; writes results JSON and the small committed weight files."""
    masks = subsets(acts)
    counts = {
        k: {"n": int(m.sum()), "deceptive": int(acts.labels[m].sum())} for k, m in masks.items()
    }
    log.info("subset sizes: %s", counts)
    sweep = layer_sweep(acts, masks, cfg, l2=1e-2)
    choice = choose(sweep, acts, masks, cfg)
    matrix = {
        kind: generalisation(acts, masks, choice, cfg, kind) for kind in ("logistic", "diff_means")
    }
    final, exported = final_probe_report(acts, masks, choice, cfg)
    n_layers = acts.mean.shape[1]
    band = [
        layer for layer in range(choice.layer, choice.layer + cfg.steering.band) if layer < n_layers
    ]
    steer = steering_vectors(acts, masks, band)
    results = {
        "model": cfg.model.name,
        "counts": counts,
        "choice": {"layer": choice.layer, "pooling": choice.pooling, "l2": choice.l2},
        "sweep": sweep,
        "generalisation": matrix,
        "final": final,
        "steering_band": band,
        "steering_gap_norms": {str(k): float(np.linalg.norm(v)) for k, v in steer.items()},
    }
    out_dir = cfg.probes.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probes.json").write_text(json.dumps(results) + "\n")
    export_weights(cfg.probes.weights_dir, cfg.model.name, choice, exported, steer)
    return results


def export_weights(
    out_dir: Path,
    model: str,
    choice: Choice,
    exported: dict[str, Any],
    steer: dict[int, np.ndarray],
) -> None:
    """``probe.safetensors`` (directions) + ``probe.json`` (biases, calibration, thresholds)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        f"{kind}.w": torch.as_tensor(v["probe"].w, dtype=torch.float32)
        for kind, v in exported.items()
    }
    tensors |= {
        f"steer.{layer}": torch.as_tensor(vec, dtype=torch.float32) for layer, vec in steer.items()
    }
    save_file(tensors, str(out_dir / "probe.safetensors"))
    meta = {
        "model": model,
        "layer": choice.layer,
        "pooling": choice.pooling,
        "l2": choice.l2,
        "steer_layers": sorted(steer),
        "probes": {
            kind: {
                "b": v["probe"].b,
                "platt_a": v["platt"].a,
                "platt_c": v["platt"].c,
                "threshold_logit": v["threshold"],
            }
            for kind, v in exported.items()
        },
    }
    (out_dir / "probe.json").write_text(json.dumps(meta, indent=2) + "\n")
