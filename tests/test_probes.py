"""Probe math: fitting, the mean-pooling identity, calibration and metrics."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from marionette.probes import (
    auroc,
    brier,
    expected_calibration_error,
    fit_diff_means,
    fit_logistic,
    fit_platt,
    positive_rate,
    threshold_at_fpr,
)


def separable(
    n: int = 200, d: int = 16, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=d)
    direction /= np.linalg.norm(direction)
    y = rng.integers(0, 2, size=n)
    x = rng.normal(size=(n, d)) + np.outer(2 * y - 1, direction) * 2.0 + 5.0
    return x, y, direction


@pytest.mark.parametrize("fit", [lambda x, y: fit_logistic(x, y, 1e-2), fit_diff_means])
def test_probes_separate_and_align_with_true_direction(fit) -> None:  # type: ignore[no-untyped-def]
    x, y, direction = separable()
    probe = fit(x, y)
    assert auroc(probe.logits(x), y) > 0.97
    assert float(probe.direction @ direction) > 0.8


def test_logistic_probe_is_affine_so_mean_pooling_commutes() -> None:
    """probe(mean of tokens) == mean of probe(token): the identity the live needle relies on."""
    x, y, _ = separable()
    probe = fit_logistic(x, y, 1e-2)
    tokens = np.random.default_rng(1).normal(size=(37, x.shape[1]))
    assert probe.logits(tokens.mean(0, keepdims=True))[0] == pytest.approx(
        probe.logits(tokens).mean(), rel=1e-9, abs=1e-9
    )


def test_fit_rejects_single_class() -> None:
    x = np.zeros((4, 3))
    with pytest.raises(ValueError, match="both classes"):
        fit_logistic(x, np.zeros(4, dtype=int), 1e-2)


def brute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


@settings(max_examples=60, deadline=None)
@given(st.lists(st.tuples(st.integers(0, 5), st.booleans()), min_size=2, max_size=40))
def test_auroc_matches_pairwise_definition_with_ties(pairs: list[tuple[int, bool]]) -> None:
    scores = np.array([float(s) for s, _ in pairs])
    labels = np.array([int(b) for _, b in pairs])
    if len(set(labels.tolist())) < 2:
        with pytest.raises(ValueError):
            auroc(scores, labels)
        return
    assert auroc(scores, labels) == pytest.approx(brute_auroc(scores, labels))


@settings(max_examples=60, deadline=None)
@given(
    st.lists(st.floats(-100, 100, allow_nan=False), min_size=1, max_size=200), st.floats(0.0, 0.5)
)
def test_threshold_respects_fpr_budget(neg: list[float], fpr: float) -> None:
    tau = threshold_at_fpr(np.array(neg), fpr)
    assert positive_rate(np.array(neg), tau) <= fpr + 1e-12


def test_platt_is_monotone_and_calibrates_logistic_data() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=4000) * 3
    y = (rng.random(4000) < 1 / (1 + np.exp(-(0.5 * z - 0.3)))).astype(int)
    platt = fit_platt(z, y)
    assert platt.a == pytest.approx(0.5, abs=0.08)
    assert platt.c == pytest.approx(-0.3, abs=0.12)
    p = platt(np.sort(z))
    assert np.all(np.diff(p) >= 0)
    assert expected_calibration_error(platt(z), y) < 0.03


def test_calibration_metrics_on_perfect_and_worst_predictions() -> None:
    y = np.array([0, 1, 0, 1])
    assert brier(y.astype(float), y) == 0
    assert expected_calibration_error(y.astype(float), y) == 0
    assert brier(1 - y.astype(float), y) == 1
