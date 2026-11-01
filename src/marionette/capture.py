"""Capture response-token activations at every layer, pooled per response (mean and max).

Pooling happens inside the forward hooks, so only ``2 x n_layers x d_model`` numbers per example
ever leave the GPU. They are written to float16 memory-mapped arrays next to an index file:

    <data.dir>/acts/index.json   ids, datasets, families, splits, labels (row order)
    <data.dir>/acts/mean.npy     [n, n_layers, d_model]  mean over response tokens
    <data.dir>/acts/max.npy      [n, n_layers, d_model]  elementwise max over response tokens
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from marionette.data.schema import Example
from marionette.lm import LanguageModel, ReadFn

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActivationSet:
    """Pooled activations plus their row metadata."""

    ids: list[str]
    datasets: np.ndarray
    families: np.ndarray
    splits: np.ndarray
    labels: np.ndarray
    mean: np.ndarray
    max: np.ndarray

    def layer(self, layer: int, pooling: str) -> np.ndarray:
        """All rows at one layer as float32, shape [n, d_model]."""
        src = {"mean": self.mean, "max": self.max}[pooling]
        return np.asarray(src[:, layer, :], dtype=np.float32)


def response_mask(lengths: list[int], starts: list[int], width: int) -> torch.Tensor:
    """[batch, width] bool mask of response tokens (end-of-turn token excluded)."""
    mask = torch.zeros(len(lengths), width, dtype=torch.bool)
    for row, (n, s) in enumerate(zip(lengths, starts, strict=True)):
        mask[row, s : n - 1] = True
    return mask


def pooling_reads(
    mask: torch.Tensor, n_layers: int, sums: torch.Tensor, maxes: torch.Tensor
) -> dict[int, ReadFn]:
    """Hooks that write masked mean and max of each layer's residual into ``sums``/``maxes``."""
    counts = mask.sum(dim=1, keepdim=True).clamp(min=1).float()

    def make(layer: int) -> ReadFn:
        def read(hidden: torch.Tensor) -> None:
            m = mask.to(hidden.device)
            h = hidden.float()
            sums[:, layer] = (h * m.unsqueeze(-1)).sum(dim=1) / counts.to(h.device)
            maxes[:, layer] = h.masked_fill(~m.unsqueeze(-1), float("-inf")).amax(dim=1)

        return read

    return {layer: make(layer) for layer in range(n_layers)}


@torch.inference_mode()
def capture(
    lm: LanguageModel, examples: list[Example], out_dir: Path, batch_size: int, max_len: int
) -> None:
    """Pool activations for every labelled, kept example and write them to ``out_dir``."""
    rows = [e for e in examples if e.kept and e.label is not None and e.response]
    encoded = [lm.encode_dialog(e.messages, e.response or "") for e in rows]
    too_long = [e.id for e, (ids, _) in zip(rows, encoded, strict=True) if len(ids) > max_len]
    if too_long:
        raise ValueError(f"{len(too_long)} dialogs exceed max_seq_len={max_len}: {too_long[:3]}")
    n, L, d = len(rows), lm.n_layers, lm.d_model
    out_dir.mkdir(parents=True, exist_ok=True)
    mean = np.lib.format.open_memmap(out_dir / "mean.npy", "w+", np.float16, (n, L, d))
    mx = np.lib.format.open_memmap(out_dir / "max.npy", "w+", np.float16, (n, L, d))
    order = sorted(range(n), key=lambda i: len(encoded[i][0]))
    pad = lm.tokenizer.pad_token_id
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        lengths = [len(encoded[i][0]) for i in idx]
        width = max(lengths)
        ids = torch.full((len(idx), width), pad, dtype=torch.long)
        attn = torch.zeros(len(idx), width, dtype=torch.long)
        for row, i in enumerate(idx):
            ids[row, : lengths[row]] = torch.tensor(encoded[i][0])
            attn[row, : lengths[row]] = 1
        mask = response_mask(lengths, [encoded[i][1] for i in idx], width)
        sums = torch.empty(len(idx), L, d, device=lm.device)
        maxes = torch.empty(len(idx), L, d, device=lm.device)
        with lm.hooks(reads=pooling_reads(mask, L, sums, maxes)):
            lm.model.model(
                input_ids=ids.to(lm.device), attention_mask=attn.to(lm.device), use_cache=False
            )
        mean[idx] = sums.cpu().numpy().astype(np.float16)
        mx[idx] = maxes.cpu().numpy().astype(np.float16)
        if (start // batch_size) % 20 == 0:
            log.info("captured %d / %d", min(start + batch_size, n), n)
    mean.flush()
    mx.flush()
    index = {
        "model": lm.name,
        "n_layers": L,
        "d_model": d,
        "ids": [e.id for e in rows],
        "datasets": [e.dataset for e in rows],
        "families": [e.family for e in rows],
        "splits": [e.split for e in rows],
        "labels": [int(e.label or 0) for e in rows],
    }
    (out_dir / "index.json").write_text(json.dumps(index))


def load_activations(out_dir: Path) -> ActivationSet:
    path = out_dir / "index.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `marionette capture` first")
    index: dict[str, Any] = json.loads(path.read_text())
    return ActivationSet(
        ids=index["ids"],
        datasets=np.array(index["datasets"]),
        families=np.array(index["families"]),
        splits=np.array(index["splits"]),
        labels=np.array(index["labels"], dtype=np.int64),
        mean=np.load(out_dir / "mean.npy", mmap_mode="r"),
        max=np.load(out_dir / "max.npy", mmap_mode="r"),
    )
