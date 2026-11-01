"""Linear deception probes, Platt calibration and the metrics used to judge them.

Both probe kinds end up as an affine map ``x -> x @ w + b`` in *raw* residual space. Logistic
regression is fitted on standardised features and then folded back, so every probe is a single
direction plus a bias. That makes a useful identity exact: the probe's logit on a mean-pooled
response equals the mean of its per-token logits, which is what lets the live app show per-token
heat that averages to the response-level verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class LinearProbe:
    """``logit(x) = x @ w + b``; higher means more deceptive."""

    w: np.ndarray
    b: float
    kind: str

    def logits(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=np.float64) @ self.w.astype(np.float64) + self.b

    @property
    def direction(self) -> np.ndarray:
        """Unit vector along ``w``."""
        return self.w / max(float(np.linalg.norm(self.w)), 1e-12)


def _check_xy(x: np.ndarray, y: np.ndarray) -> None:
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError(f"expected x [n, d] and y [n], got {x.shape} and {y.shape}")
    classes = set(np.unique(y).tolist())
    if classes != {0, 1}:
        raise ValueError(f"need both classes 0 and 1 to fit a probe, got {sorted(classes)}")


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float, max_iter: int = 300) -> LinearProbe:
    """L2-regularised logistic regression on standardised features (full-batch L-BFGS).

    The objective is ``mean BCE + l2/2 * ||v||^2`` over standardised coefficients ``v``; the
    returned probe has been mapped back to raw-activation coordinates.
    """
    _check_xy(x, y)
    xt = torch.as_tensor(x, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.float64)
    mu = xt.mean(0)
    sigma = xt.std(0).clamp(min=1e-6)
    z = (xt - mu) / sigma
    v = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    c = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS(
        [v, c],
        lr=1.0,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
    )

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z @ v + c, yt)
        loss = loss + 0.5 * l2 * (v * v).sum()
        loss.backward()  # type: ignore[no-untyped-call]
        return loss

    opt.step(closure)  # type: ignore[no-untyped-call]
    with torch.no_grad():
        w = (v / sigma).numpy()
        b = float(c - (v * mu / sigma).sum())
    return LinearProbe(w=w, b=b, kind="logistic")


def fit_diff_means(x: np.ndarray, y: np.ndarray) -> LinearProbe:
    """Difference of class means; the threshold sits halfway between the two means."""
    _check_xy(x, y)
    x64 = np.asarray(x, dtype=np.float64)
    mu1, mu0 = x64[y == 1].mean(0), x64[y == 0].mean(0)
    w = mu1 - mu0
    return LinearProbe(w=w, b=-float(w @ (mu1 + mu0) / 2), kind="diff_means")


@dataclass(frozen=True)
class PlattCalibrator:
    """``p = sigmoid(a * logit + c)``, fitted on held-out logits."""

    a: float
    c: float

    def __call__(self, logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self.a * np.asarray(logits, dtype=np.float64) + self.c)))


def fit_platt(logits: np.ndarray, y: np.ndarray) -> PlattCalibrator:
    _check_xy(np.asarray(logits).reshape(-1, 1), y)
    s = torch.as_tensor(logits, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.float64)
    scale = float(s.std().clamp(min=1e-6))
    a = torch.tensor(1.0 / scale, dtype=torch.float64, requires_grad=True)
    c = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([a, c], lr=1.0, max_iter=200, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(a * s + c, yt)
        loss.backward()  # type: ignore[no-untyped-call]
        return loss

    opt.step(closure)  # type: ignore[no-untyped-call]
    return PlattCalibrator(a=float(a.detach()), c=float(c.detach()))


# ---- metrics -------------------------------------------------------------------------------


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (ties count one half)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("AUROC needs at least one positive and one negative")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # Average the ranks of tied scores.
    allscores = np.concatenate([pos, neg])
    _, inverse, counts = np.unique(allscores, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    ranks = (sums / counts)[inverse]
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def threshold_at_fpr(negative_scores: np.ndarray, fpr: float) -> float:
    """Smallest threshold whose false-positive rate on ``negative_scores`` is at most ``fpr``.

    A score counts as positive when it is strictly greater than the threshold.
    """
    neg = np.sort(np.asarray(negative_scores, dtype=np.float64))
    if len(neg) == 0:
        raise ValueError("need negative scores to set a threshold")
    allowed = int(np.floor(fpr * len(neg)))
    return float(neg[len(neg) - 1 - allowed])


def positive_rate(scores: np.ndarray, threshold: float) -> float:
    s = np.asarray(scores, dtype=np.float64)
    return float((s > threshold).mean()) if len(s) else float("nan")


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Mean |accuracy - confidence| over equal-width probability bins, weighted by bin size."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    err = 0.0
    for b in range(bins):
        sel = which == b
        if sel.any():
            err += sel.mean() * abs(y[sel].mean() - p[sel].mean())
    return float(err)


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((np.asarray(probs, dtype=np.float64) - np.asarray(labels)) ** 2))
