"""Batched, seeded chat generation (optionally under steering) and single-token judging."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

import torch

from marionette.data.schema import Message
from marionette.lm import EditFn, LanguageModel

log = logging.getLogger(__name__)


def _left_padded(lm: LanguageModel, texts: Sequence[str]) -> dict[str, torch.Tensor]:
    tok = lm.tokenizer
    tok.padding_side = "left"
    batch = tok(list(texts), add_special_tokens=False, padding=True, return_tensors="pt")
    return {k: v.to(lm.device) for k, v in batch.items()}


@torch.inference_mode()
def generate_replies(
    lm: LanguageModel,
    prompts: Sequence[Sequence[Message]],
    *,
    max_new_tokens: int,
    batch_size: int,
    temperature: float,
    top_p: float,
    seed: int,
    edits: Mapping[int, EditFn] | None = None,
) -> list[str]:
    """One reply per prompt. Prompts are length-sorted into batches to minimise padding.

    Each batch reseeds the sampler from ``seed`` and its first prompt index, so a reply does not
    depend on which other prompts were generated before it in the same process.
    """
    texts = [lm.render_prompt(p) for p in prompts]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    replies: list[str] = [""] * len(texts)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        batch = _left_padded(lm, [texts[i] for i in idx])
        torch.manual_seed(seed + idx[0])
        sampling = temperature > 0
        with lm.hooks(edits=edits):
            out = lm.model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=sampling,
                temperature=temperature if sampling else None,
                top_p=top_p if sampling else None,
                top_k=None,
                eos_token_id=[lm.end_of_turn_id, lm.tokenizer.eos_token_id],
                pad_token_id=lm.tokenizer.pad_token_id,
            )
        new = out[:, batch["input_ids"].shape[1] :]
        for row, i in enumerate(idx):
            replies[i] = lm.tokenizer.decode(new[row], skip_special_tokens=True).strip()
        log.info("generated %d / %d", min(start + batch_size, len(order)), len(order))
    return replies


@torch.inference_mode()
def choice_probabilities(
    lm: LanguageModel,
    prompts: Sequence[Sequence[Message]],
    choices: Sequence[str],
    batch_size: int,
) -> torch.Tensor:
    """P(first reply token = each choice), renormalised over ``choices``. Shape [n, n_choices].

    Reading one forward pass's logits is deterministic and far cheaper than sampling a verdict.
    """
    choice_ids = []
    for c in choices:
        ids = lm.tokenizer(c, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"choice {c!r} is not a single token ({ids})")
        choice_ids.append(ids[0])
    texts = [lm.render_prompt(p) for p in prompts]
    probs = torch.empty(len(texts), len(choices))
    for start in range(0, len(texts), batch_size):
        batch = _left_padded(lm, texts[start : start + batch_size])
        # logits_to_keep=1: only the last position's logits, never a [batch, seq, vocab] tensor.
        logits = lm.model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
        probs[start : start + len(batch["input_ids"])] = torch.softmax(
            logits[:, choice_ids], dim=-1
        ).cpu()
    return probs
