"""SAE feature analysis: which latents drive the deception probe, and what do they mean?

1. **Alignment.** For every labelled training response we take mean SAE activations over the
   response tokens. Because the probe is linear, the SAE reconstruction's probe logit splits
   exactly into per-latent terms ``f_i * (d_i . w)``; a latent's *contribution* is the mean of that
   term on deceptive minus honest responses. We also report each decoder row's cosine with the
   truth-serum direction.
2. **Top activations.** Max-activating contexts per latent over held-out corpus text plus the
   dataset dialogs (so deception-specific latents have relevant examples).
3. **Auto-labels.** The studied model reads a latent's top contexts (activating token marked)
   and names the concept. **Detection scoring** checks the label: the model sees held-out
   snippets (some activating, some not) and gives P(yes, this matches). AUROC of P(yes) against
   the truth is compared with a baseline where each latent is scored with a *shuffled* label.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from marionette.config import Config
from marionette.corpus import open_tokens
from marionette.data.schema import Example, Message
from marionette.generation import choice_probabilities, generate_replies
from marionette.lm import LanguageModel
from marionette.probes import auroc
from marionette.sae import BatchTopKSAE
from marionette.sae_train import keep_mask, token_paths
from marionette.steering import ProbeBundle

log = logging.getLogger(__name__)


@dataclass
class TopActivations:
    """Per latent: the ``k`` largest per-context max activations, with where they happened."""

    values: torch.Tensor  # [k, n_latents]
    context: torch.Tensor  # [k, n_latents] row into the context list
    position: torch.Tensor  # [k, n_latents]

    @classmethod
    def empty(cls, k: int, n: int, device: torch.device) -> TopActivations:
        return cls(
            torch.zeros(k, n, device=device),
            torch.full((k, n), -1, device=device, dtype=torch.long),
            torch.zeros(k, n, device=device, dtype=torch.long),
        )

    def update(self, acts: torch.Tensor, mask: torch.Tensor, first_context: int) -> None:
        """Merge a batch of activations ``[batch, seq, n_latents]`` (masked positions ignored)."""
        acts = acts.masked_fill(~mask.unsqueeze(-1), 0.0)
        best, pos = acts.max(dim=1)  # [batch, n]
        ctx = (
            torch.arange(first_context, first_context + acts.shape[0], device=acts.device)
            .unsqueeze(1)
            .expand_as(best)
        )
        values = torch.cat([self.values, best])
        k = self.values.shape[0]
        top, idx = values.topk(k, dim=0)
        self.context = torch.cat([self.context, ctx]).gather(0, idx)
        self.position = torch.cat([self.position, pos]).gather(0, idx)
        self.values = top


def dialog_contexts(lm: LanguageModel, examples: list[Example], max_len: int) -> list[list[int]]:
    rows = [e for e in examples if e.kept and e.response]
    out = []
    for e in rows:
        ids, _ = lm.encode_dialog(e.messages, e.response or "")
        if len(ids) <= max_len:
            out.append(ids)
    return out


@torch.inference_mode()
def collect(
    lm: LanguageModel, sae: BatchTopKSAE, contexts: list[list[int]], k: int, batch: int = 16
) -> tuple[TopActivations, torch.Tensor]:
    """Top activations per latent and how often each latent fires, over ``contexts``."""
    n = sae.spec.n_latents
    top = TopActivations.empty(k, n, lm.device)
    fires = torch.zeros(n, device=lm.device)
    n_tok = 0
    pad = lm.tokenizer.pad_token_id
    for start in range(0, len(contexts), batch):
        chunk = contexts[start : start + batch]
        width = max(len(c) for c in chunk)
        ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        attn = torch.zeros_like(ids)
        for row, c in enumerate(chunk):
            ids[row, : len(c)] = torch.tensor(c)
            attn[row, : len(c)] = 1
        ids, attn = ids.to(lm.device), attn.to(lm.device)
        h = lm.run_until(ids, attn, [sae.spec.layer])[sae.spec.layer]
        mask = keep_mask(lm, ids) & attn.bool()
        f = sae.encode(h.float())
        top.update(f, mask, start)
        fires += ((f > 0) & mask.unsqueeze(-1)).sum((0, 1)).float()
        n_tok += int(mask.sum())
    return top, fires / max(n_tok, 1)


@torch.inference_mode()
def response_means(lm: LanguageModel, sae: BatchTopKSAE, examples: list[Example]) -> torch.Tensor:
    """Mean SAE activation over each example's response tokens, ``[n_examples, n_latents]``."""
    out = torch.zeros(len(examples), sae.spec.n_latents)
    for i, e in enumerate(examples):
        ids, start = lm.encode_dialog(e.messages, e.response or "")
        t = torch.tensor([ids], device=lm.device)
        h = lm.run_until(t, None, [sae.spec.layer])[sae.spec.layer][0, start:-1]
        out[i] = sae.encode(h.float()).mean(0).cpu()
    return out


def alignment(
    sae: BatchTopKSAE, bundle: ProbeBundle, means: torch.Tensor, labels: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Per-latent probe contribution (deceptive minus honest), activation AUROC and cosine."""
    dirs = sae.feature_directions().detach().float().cpu()
    # f_i * W_dec[i] / input_scale is latent i's term in the reconstruction (raw units).
    dec = sae.W_dec.detach().float().cpu() / float(sae.input_scale)
    probe_w = dec @ bundle.w.float()
    contrib = means * probe_w
    diff = contrib[labels == 1].mean(0) - contrib[labels == 0].mean(0)
    steer = torch.stack([v.float() for v in bundle.steer.values()]).mean(0)
    cos = dirs @ (steer / steer.norm())
    return {"contribution": diff, "cosine_with_serum": cos, "probe_weight": probe_w}


def snippet(
    lm: LanguageModel, ids: list[int], pos: int, before: int, after: int, mark: bool
) -> str:
    lo = max(1, pos - before)
    left = lm.tokenizer.decode(ids[lo:pos], skip_special_tokens=True)
    tok = lm.tokenizer.decode(ids[pos : pos + 1], skip_special_tokens=True)
    right = lm.tokenizer.decode(ids[pos + 1 : pos + 1 + after], skip_special_tokens=True)
    text = f"{left}[[{tok}]]{right}" if mark else f"{left}{tok}{right}"
    return " ".join(text.split())


LABEL_PROMPT = (
    "Below are text snippets where one internal feature of a language model is strongly active."
    " The token where it fires most is marked like [[this]].\n\n{examples}\n\nIn at most six words,"
    " what concept does this feature detect? Reply with the label only."
)
SCORE_PROMPT = (
    'Feature description: {label}\n\nText: "{text}"\n\nDoes the text contain something that'
    " matches the feature description? Answer Yes or No."
)


def label_features(
    cfg: Config, lm: LanguageModel, examples_per_feature: list[list[str]]
) -> list[str]:
    prompts = [
        [
            Message(
                role="user",
                content=LABEL_PROMPT.format(
                    examples="\n".join(f"{i + 1}. {s}" for i, s in enumerate(snips))
                ),
            )
        ]
        for snips in examples_per_feature
    ]
    labels = generate_replies(
        lm,
        prompts,
        max_new_tokens=cfg.autointerp.max_label_tokens,
        batch_size=cfg.autointerp.batch_size,
        temperature=0.0,
        top_p=1.0,
        seed=cfg.seed,
    )
    return [
        lab.strip().strip('."').splitlines()[0][:60] if lab.strip() else "(no label)"
        for lab in labels
    ]


def detection_scores(
    cfg: Config,
    lm: LanguageModel,
    labels: list[str],
    positives: list[list[str]],
    negatives: list[list[str]],
) -> list[float | None]:
    """AUROC of the model's P(yes) at telling activating from non-activating snippets."""
    prompts: list[list[Message]] = []
    spans = []
    for label, pos, neg in zip(labels, positives, negatives, strict=True):
        spans.append((len(prompts), len(pos), len(neg)))
        prompts += [
            [Message(role="user", content=SCORE_PROMPT.format(label=label, text=t))]
            for t in pos + neg
        ]
    probs = choice_probabilities(lm, prompts, ("Yes", "No"), cfg.autointerp.batch_size * 2)
    yes = probs[:, 0].numpy()
    out: list[float | None] = []
    for start, n_pos, n_neg in spans:
        if n_pos == 0 or n_neg == 0:
            out.append(None)
            continue
        y = np.array([1] * n_pos + [0] * n_neg)
        out.append(auroc(yes[start : start + n_pos + n_neg], y))
    return out


def run(
    cfg: Config, lm: LanguageModel, sae: BatchTopKSAE, bundle: ProbeBundle, examples: list[Example]
) -> dict[str, Any]:
    """Alignment, top activations, labels and detection scores; writes ``features.json``."""
    ai = cfg.autointerp
    rng = random.Random(cfg.seed)
    train = [
        e
        for e in examples
        if e.kept
        and e.label is not None
        and e.split == "train"
        and e.dataset in ("instructed", "roleplay")
    ]
    means = response_means(lm, sae, train)
    labels_t = torch.tensor([e.label for e in train])
    align = alignment(sae, bundle, means, labels_t)
    deception_auroc = _activation_auroc(means, labels_t)

    corpus = [
        list(map(int, row)) for row in open_tokens(token_paths(cfg)[1])[: ai.collect_contexts]
    ]
    dialogs = dialog_contexts(lm, examples, cfg.data.max_seq_len)
    contexts = corpus + dialogs
    top, freq = collect(lm, sae, contexts, ai.top_examples)
    dialog_freq = collect(lm, sae, dialogs, 1)[1]

    contrib = align["contribution"]
    chosen = [int(i) for i in contrib.topk(ai.n_deception).indices]
    by_freq = dialog_freq.cpu().argsort(descending=True).tolist()
    for i in by_freq:
        if len(chosen) >= ai.n_features:
            break
        if i not in chosen and float(top.values[0, i]) > 0:
            chosen.append(int(i))
    chosen = [i for i in chosen if float(top.values[0, i]) > 0]

    label_snips, pos_snips, neg_snips = [], [], []
    for i in chosen:
        ranked = [
            (int(top.context[r, i]), int(top.position[r, i]))
            for r in range(ai.top_examples)
            if float(top.values[r, i]) > 0
        ]
        label_snips.append(
            [
                snippet(lm, contexts[c], p, ai.window_before, ai.window_after, True)
                for c, p in ranked[: ai.label_examples]
            ]
        )
        held = ranked[ai.label_examples :][:4]
        pos_snips.append(
            [snippet(lm, contexts[c], p, ai.window_before, ai.window_after, False) for c, p in held]
        )
        active_ctx = {c for c, _ in ranked}
        neg: list[str] = []
        while len(neg) < len(held):
            c = rng.randrange(len(corpus))
            if c not in active_ctx:
                p = rng.randrange(1, len(contexts[c]))
                neg.append(snippet(lm, contexts[c], p, ai.window_before, ai.window_after, False))
        neg_snips.append(neg)
    labels = label_features(cfg, lm, label_snips)
    scores = detection_scores(cfg, lm, labels, pos_snips, neg_snips)
    shuffled = labels[:]
    rng.shuffle(shuffled)
    baseline = detection_scores(cfg, lm, shuffled, pos_snips, neg_snips)

    features = []
    for j, i in enumerate(chosen):
        features.append(
            {
                "id": i,
                "label": labels[j],
                "detection_auroc": scores[j],
                "shuffled_auroc": baseline[j],
                "deception_contribution": float(contrib[i]),
                "deception_auroc": deception_auroc[i],
                "cosine_with_serum": float(align["cosine_with_serum"][i]),
                "density": float(freq[i]),
                "max_activation": float(top.values[0, i]),
                "examples": label_snips[j][:3],
            }
        )
    valid = [
        (f["detection_auroc"], f["shuffled_auroc"])
        for f in features
        if f["detection_auroc"] is not None and f["shuffled_auroc"] is not None
    ]
    summary = {
        "n_labelled": len(features),
        "mean_detection_auroc": float(np.mean([a for a, _ in valid])) if valid else None,
        "mean_shuffled_auroc": float(np.mean([b for _, b in valid])) if valid else None,
        "n_scored": len(valid),
        "dead_or_silent_in_collection": int((top.values[0] == 0).sum()),
    }
    result = {
        "model": cfg.model.name,
        "layer": sae.spec.layer,
        "summary": summary,
        "features": features,
    }
    cfg.probes.weights_dir.mkdir(parents=True, exist_ok=True)
    (cfg.probes.weights_dir / "features.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def _activation_auroc(means: torch.Tensor, labels: torch.Tensor) -> list[float | None]:
    """AUROC of each latent's mean response activation as a lie detector on its own."""
    y = labels.numpy()
    m = means.numpy()
    active = (m > 0).any(0)
    return [auroc(m[:, i], y) if active[i] else None for i in range(m.shape[1])]
