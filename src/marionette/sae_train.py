"""SAE stages: token preparation, streamed training, and held-out evaluation.

Activations are produced on the fly by the (truncated) language model into a GPU shuffle
buffer and consumed by the SAE; none are written to disk.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from marionette.config import Config
from marionette.corpus import (
    Source,
    iter_documents,
    make_formatter,
    mix_contexts,
    open_tokens,
    pack_contexts,
    tokenize_documents,
    write_token_file,
)
from marionette.lm import LanguageModel
from marionette.sae import BatchTopKSAE, SAESpec

log = logging.getLogger(__name__)

CONTEXT_PREFIX = "<|endoftext|>"


def token_paths(cfg: Config) -> tuple[Path, Path]:
    return cfg.sae.dir / "train_tokens.npy", cfg.sae.dir / "heldout_tokens.npy"


def prepare_tokens(cfg: Config, tokenizer: Any) -> None:
    """Tokenise a FineWeb-Edu / UltraChat mixture into train and held-out context files."""
    s = cfg.sae
    prefix_id = int(tokenizer.convert_tokens_to_ids(CONTEXT_PREFIX))
    sources = [
        Source(s.fineweb_repo, tuple(s.fineweb_files), "text", "text"),
        Source(s.chat_repo, tuple(s.chat_files), "messages", "chat"),
    ]
    streams = [
        pack_contexts(
            tokenize_documents(
                iter_documents(src, make_formatter(src.kind, tokenizer)), tokenizer, prefix_id
            ),
            s.ctx_len,
            prefix_id,
        )
        for src in sources
    ]
    mixed = mix_contexts(streams, [1 - s.chat_fraction, s.chat_fraction], s.seed)
    train_path, heldout_path = token_paths(cfg)
    # Held-out contexts come later in the same streams, so they are different documents.
    write_token_file(train_path, mixed, math.ceil(s.train_tokens / s.ctx_len), s.ctx_len)
    write_token_file(heldout_path, mixed, math.ceil(s.heldout_tokens / s.ctx_len), s.ctx_len)


def keep_mask(lm: LanguageModel, ids: torch.Tensor) -> torch.Tensor:
    """Positions the SAE sees: not position 0 (attention sink) and not special tokens.

    Both carry residual norms far above typical tokens and would dominate an MSE objective
    while carrying no semantic content.
    """
    special = torch.tensor(
        sorted(set(lm.tokenizer.all_special_ids) | _added_special(lm)), device=ids.device
    )
    mask = ~torch.isin(ids, special)
    mask[:, 0] = False
    return mask


def _added_special(lm: LanguageModel) -> set[int]:
    decoder = getattr(lm.tokenizer, "added_tokens_decoder", {})
    return {int(i) for i, tok in decoder.items() if getattr(tok, "special", False)}


def activation_batches(
    lm: LanguageModel, tokens: np.ndarray, layer: int, cfg: Config
) -> Iterator[torch.Tensor]:
    """Endless-until-exhausted shuffled batches of residuals from a GPU shuffle buffer."""
    s = cfg.sae
    gen = torch.Generator(device=lm.device).manual_seed(s.seed)
    buf = torch.empty(s.buffer_tokens, lm.d_model, dtype=torch.bfloat16, device=lm.device)
    fill = 0
    ctx = 0

    def refill() -> bool:
        nonlocal fill, ctx
        while fill < s.buffer_tokens and ctx < len(tokens):
            ids = torch.from_numpy(tokens[ctx : ctx + s.lm_batch_contexts].astype(np.int64))
            ctx += s.lm_batch_contexts
            ids = ids.to(lm.device)
            h = lm.run_until(ids, None, [layer])[layer]
            acts = h[keep_mask(lm, ids)]
            take = min(len(acts), s.buffer_tokens - fill)
            buf[fill : fill + take] = acts[:take]
            fill += take
        return fill >= s.batch_tokens

    while refill():
        perm = torch.randperm(fill, device=lm.device, generator=gen)
        buf[:fill] = buf[perm]
        # Serve half the buffer, then refill it, so batches mix tokens from many contexts.
        half = max(s.batch_tokens, fill // 2)
        n_batches = half // s.batch_tokens
        for i in range(n_batches):
            yield buf[fill - (i + 1) * s.batch_tokens : fill - i * s.batch_tokens].float()
        fill -= n_batches * s.batch_tokens


def lr_at(step: int, total: int, cfg: Config) -> float:
    s = cfg.sae
    if step < s.warmup_steps:
        return s.lr * (step + 1) / max(1, s.warmup_steps)
    decay_from = int(total * s.decay_start)
    if step >= decay_from:
        return s.lr * max(0.0, (total - step) / max(1, total - decay_from))
    return s.lr


def train(cfg: Config, lm: LanguageModel, layer: int) -> dict[str, Any]:
    """Train one SAE on the residual stream after block ``layer``; saves it under ``sae.dir``."""
    s = cfg.sae
    tokens = open_tokens(token_paths(cfg)[0])
    spec = SAESpec(
        d_in=lm.d_model, n_latents=s.n_latents, k=s.k, aux_k=s.aux_k, layer=layer, model=lm.name
    )
    sae = BatchTopKSAE(spec, seed=s.seed).to(lm.device)
    # ~5% of positions are special tokens; 0.9 keeps the LR schedule ending before data does.
    available = int(tokens.shape[0] * (tokens.shape[1] - 1) * 0.9) // s.batch_tokens
    total = min(available, s.train_tokens // s.batch_tokens)
    batches = activation_batches(lm, tokens, layer, cfg)
    first = next(batches)
    with torch.no_grad():
        sae.input_scale.fill_(math.sqrt(lm.d_model) / float(first.norm(dim=-1).mean()))
        sae.b_dec.copy_((first * sae.input_scale).mean(0))
    opt = torch.optim.Adam(sae.parameters(), lr=s.lr, betas=(0.9, 0.999), eps=6.25e-10)
    history: list[dict[str, float]] = []
    t0 = time.time()
    step = 0
    batch: torch.Tensor | None = first
    while batch is not None and step < total:
        for group in opt.param_groups:
            group["lr"] = lr_at(step, total, cfg)
        with torch.autocast(
            device_type=lm.device.type, dtype=torch.bfloat16, enabled=lm.device.type == "cuda"
        ):
            out = sae.training_loss(batch * sae.input_scale, s.aux_coef, s.dead_tokens)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()  # type: ignore[no-untyped-call]
        sae.remove_parallel_decoder_grad()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        sae.normalise_decoder()
        sae.update_threshold(out["min_kept"], s.threshold_beta)
        if step % s.log_every == 0:
            row = {k: float(v) for k, v in out.items() if k != "loss"}
            row |= {
                "step": step,
                "tokens": (step + 1) * s.batch_tokens,
                "elapsed_s": time.time() - t0,
            }
            history.append(row)
            log.info(
                "step %d/%d fvu %.4f l0 %.1f dead %.3f aux %.4f (%.0fs)",
                step,
                total,
                row["fvu"],
                row["l0"],
                row["dead_fraction"],
                row["aux"],
                row["elapsed_s"],
            )
        step += 1
        batch = next(batches, None)
    stats = {
        "steps": step,
        "tokens": step * s.batch_tokens,
        "train_seconds": time.time() - t0,
        "threshold": float(sae.threshold),
        "history": history,
    }
    sae.save(s.dir, stats)
    return stats


@torch.inference_mode()
def evaluate(
    cfg: Config, lm: LanguageModel, sae: BatchTopKSAE, n_contexts: int = 256
) -> dict[str, Any]:
    """FVU, L0, dead fraction and CE loss recovered on held-out contexts (full model needed)."""
    tokens = open_tokens(token_paths(cfg)[1])[:n_contexts]
    layer = sae.spec.layer
    sse = tot = l0_sum = n_tok = 0.0
    ce = {"clean": 0.0, "sae": 0.0, "zero": 0.0}
    fired = torch.zeros(sae.spec.n_latents, dtype=torch.bool, device=lm.device)
    mean_acc = torch.zeros(lm.d_model, device=lm.device, dtype=torch.float64)
    # Kept per token (on CPU) because a few very-high-norm tokens can dominate the global FVU.
    xs: list[torch.Tensor] = []
    errs: list[torch.Tensor] = []
    for start in range(0, len(tokens), 8):
        ids = torch.from_numpy(tokens[start : start + 8].astype(np.int64)).to(lm.device)
        mask = keep_mask(lm, ids)
        h = lm.run_until(ids, None, [layer])[layer]
        x = h[mask].float()
        f = sae.encode(x)
        x_hat = sae.decode(f)
        sse += float((x - x_hat).pow(2).sum())
        xs.append(x.cpu())
        errs.append((x - x_hat).pow(2).sum(-1).cpu())
        mean_acc += x.sum(0).double()
        tot += float(x.pow(2).sum())
        l0_sum += float((f > 0).sum())
        n_tok += len(x)
        fired |= (f > 0).any(0)
        for name in ce:
            ce[name] += _ce(lm, ids, mask, layer, sae, name) * ids.shape[0]
    mean = mean_acc / n_tok
    total_var = tot - n_tok * float(mean.pow(2).sum())
    for name in ce:
        ce[name] /= len(tokens)
    recovered = (ce["zero"] - ce["sae"]) / (ce["zero"] - ce["clean"])
    return {
        "fvu": sse / total_var,
        **_per_token_fvu(torch.cat(xs), torch.cat(errs)),
        "l0": l0_sum / n_tok,
        "dead_fraction_heldout": float((~fired).float().mean()),
        "dead_fraction_training": float(
            (sae.tokens_since_fired > cfg.sae.dead_tokens).float().mean()
        ),
        "ce_clean": ce["clean"],
        "ce_sae": ce["sae"],
        "ce_zero_ablation": ce["zero"],
        "ce_delta_nats": ce["sae"] - ce["clean"],
        "ce_loss_recovered": recovered,
        "eval_tokens": int(n_tok),
        "eval_contexts": len(tokens),
    }


def _per_token_fvu(x: torch.Tensor, err: torch.Tensor) -> dict[str, float]:
    """FVU diagnostics that a handful of outlier tokens cannot dominate."""
    centred = (x - x.mean(0)).pow(2).sum(-1).double()
    norms = x.pow(2).sum(-1)
    top = norms >= torch.quantile(norms, 0.99)
    err = err.double()
    return {
        "median_token_fvu": float((err / centred.clamp(min=1e-12)).median()),
        "fvu_without_top1pct_norm_tokens": float(err[~top].sum() / centred[~top].sum()),
        "top1pct_norm_tokens_share_of_variance": float(centred[top].sum() / centred.sum()),
    }


def _ce(
    lm: LanguageModel,
    ids: torch.Tensor,
    mask: torch.Tensor,
    layer: int,
    sae: BatchTopKSAE,
    mode: str,
) -> float:
    """Mean next-token cross-entropy with the residual at ``layer`` clean, SAE-spliced or zeroed.

    Only the positions the SAE was trained on are replaced; the sink and special tokens stay.
    """

    def splice(h: torch.Tensor) -> torch.Tensor:
        new = h.clone()
        if mode == "sae":
            new[mask] = sae(h[mask].float()).to(h.dtype)
        else:
            new[mask] = 0
        return new

    edits = {} if mode == "clean" else {layer: splice}
    with lm.hooks(edits=edits):
        logits = lm.model(input_ids=ids, use_cache=False).logits[:, :-1].float()
    target = ids[:, 1:]
    return float(
        torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
    )


def write_eval(cfg: Config, stats: dict[str, Any]) -> None:
    cfg.probes.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.probes.out_dir / "sae.json").write_text(json.dumps(stats, indent=2) + "\n")
