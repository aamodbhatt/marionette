"""Render the README figures from a profile's result JSON files (CPU only).

uv run python scripts/make_plots.py --results results/qwen3-4b --out docs/assets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG, FG, MUTED, GRID = "#0b0d12", "#e6e8ee", "#8b93a7", "#1f2430"
RED, AMBER, INDIGO, CYAN, GREEN, GREY = (
    "#f43f5e",
    "#fbbf24",
    "#818cf8",
    "#22d3ee",
    "#34d399",
    "#a3a3a3",
)


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.edgecolor": "#39404f",
            "axes.labelcolor": FG,
            "text.color": FG,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "axes.grid": True,
            "font.size": 11,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def auroc_by_layer(p: dict[str, Any], out: Path) -> None:
    sw = p["sweep"]
    layers = np.arange(len(sw["logistic/mean"]["roleplay_val"]))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    series = [
        ("logistic/mean", "roleplay_test", "scripted role-play lies, held-out families", RED, "-"),
        ("logistic/mean", "spontaneous_test", "spontaneous lies, held-out families", AMBER, "-"),
        ("logistic/mean", "spontaneous_seen", "spontaneous lies, selection set", AMBER, ":"),
        ("diff_means/mean", "spontaneous_test", "diff-of-means, spontaneous held-out", CYAN, "--"),
        ("logistic/mean", "facts", "plain false statements (transfer)", INDIGO, "-"),
    ]
    for key, subset, label, color, ls in series:
        y = [np.nan if v is None else v for v in sw[key][subset]]
        ax.plot(layers, y, ls, color=color, label=label, lw=2)
    ax.axvline(p["choice"]["layer"], color=FG, alpha=0.4, ls="--", lw=1)
    ax.set(
        xlabel="layer (residual stream after block)",
        ylabel="AUROC",
        ylim=(0.4, 1.01),
        title="Scripted lies are easy everywhere; chosen lies peak mid-network",
    )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "auroc_by_layer.png", dpi=150)
    plt.close(fig)


def generalisation(p: dict[str, Any], out: Path) -> None:
    g = p["generalisation"]["logistic"]
    rows = list(g)
    cols = ["facts", "instructed", "roleplay_val", "roleplay_test", "spontaneous_test"]
    names = [
        "facts",
        "instructed",
        "role-play\n(val fams)",
        "role-play\n(test fams)",
        "spontaneous\n(test fams)",
    ]
    m = np.array([[np.nan if g[r][c] is None else g[r][c] for c in cols] for r in rows])
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    # aspect="auto" lets the cells fill the figure instead of leaving a wide empty margin.
    im = ax.imshow(m, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(cols)), names, fontsize=8.5)
    ax.set_yticks(range(len(rows)), [f"train: {r}" for r in rows])
    ax.grid(False)
    for i in range(len(rows)):
        for j in range(len(cols)):
            if not np.isnan(m[i, j]):
                ax.text(
                    j,
                    i,
                    f"{m[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="#0b0d12",
                    fontweight="bold",
                )
    ax.set_title(f"Cross-dataset AUROC (layer {p['choice']['layer']})")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out / "generalisation.png", dpi=150)
    plt.close(fig)


def steering(s: dict[str, Any], out: Path) -> None:
    rows = s["rows"]
    a = [r["alpha"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.plot(
        a,
        [r["deceptive"]["deceptive_rate"] for r in rows],
        "o-",
        color=RED,
        lw=2,
        label="deception rate, scripted liars",
    )
    ax.plot(
        a,
        [r["spontaneous"]["deceptive_rate"] for r in rows],
        "o-",
        color=AMBER,
        lw=2,
        label="deception rate, spontaneous",
    )
    ax.plot(
        a,
        [r["deceptive"]["unclear_rate"] for r in rows],
        "o--",
        color=GREY,
        lw=1.5,
        label="unclear/incoherent, scripted",
    )
    ax.plot(
        a,
        [r["qa_accuracy"] for r in rows],
        "s-",
        color=GREEN,
        lw=2,
        label="QA accuracy (capability)",
    )
    ax.set(
        xlabel="truth serum alpha (multiples of the lie-minus-honest mean gap, per layer)",
        ylabel="rate",
        ylim=(-0.02, 1.02),
        title=f"Truth serum on held-out families (layers {s['layers'][0]}-{s['layers'][-1]})",
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "steering.png", dpi=150)
    plt.close(fig)


def score_distributions(p: dict[str, Any], probe_meta: dict[str, Any], out: Path) -> None:
    sc = p["final"]["logistic"]["scores"]
    thr = probe_meta["probes"]["logistic"]["threshold_logit"]
    groups = [
        ("role-play lies (test fams)", sc["roleplay_test"], 1, RED),
        ("role-play honest (test fams)", sc["roleplay_test"], 0, GREEN),
        ("benign chat (unseen half)", sc["benign_test"], 0, CYAN),
        ("fiction about deception", sc.get("fiction"), 0, INDIGO),
        ("confessions", sc.get("confession"), 0, GREY),
        ("spontaneous lies (test fams)", sc.get("spontaneous_test"), 1, AMBER),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for label, data, cls, color in groups:
        if not data:
            continue
        vals = [z for z, y in zip(data["logit"], data["label"], strict=True) if y == cls]
        if vals:
            ax.hist(
                vals,
                bins=30,
                alpha=0.55,
                color=color,
                label=f"{label} (n={len(vals)})",
                density=True,
            )
    ax.axvline(thr, color=FG, ls="--", lw=1.2, label="threshold: 1% FPR on benign (val half)")
    ax.set(
        xlabel="probe logit (mean over response tokens)",
        ylabel="density",
        title="Where each kind of reply lands",
    )
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out / "scores.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/qwen3-4b"))
    ap.add_argument("--weights", type=Path, default=Path("weights/qwen3-4b"))
    ap.add_argument("--out", type=Path, default=Path("docs/assets"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    probes = json.loads((args.results / "probes.json").read_text())
    auroc_by_layer(probes, args.out)
    generalisation(probes, args.out)
    score_distributions(probes, json.loads((args.weights / "probe.json").read_text()), args.out)
    steering_path = args.results / "steering.json"
    if steering_path.is_file():
        steering(json.loads(steering_path.read_text()), args.out)


if __name__ == "__main__":
    main()
