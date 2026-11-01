"""Measure the truth serum: deception rate on held-out scenarios and QA accuracy versus alpha.

Prompts are fresh draws from the *test* families (a new seed, so not the probe's training or
evaluation instances). Replies are generated under steering and judged by the *unsteered* model.
Capability retention is the answer-key accuracy on held-out factual questions asked with an
honest instruction, so any drop there is damage, not honesty.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

from marionette.config import Config
from marionette.data import facts, scenarios
from marionette.data.build import judge_examples
from marionette.data.schema import Example
from marionette.generation import generate_replies
from marionette.lm import LanguageModel
from marionette.steering import ProbeBundle, truth_serum

log = logging.getLogger(__name__)

EVAL_SEED_OFFSET = 10_000


def eval_prompts(cfg: Config) -> tuple[list[Example], list[Example], list[Example]]:
    """Held-out deceptive-condition and spontaneous scenarios, plus honest QA questions."""
    seed = cfg.seed + EVAL_SEED_OFFSET
    rng = random.Random(seed)
    pool = scenarios.scenario_examples(
        cfg.steering.eval_scenarios, cfg.steering.eval_scenarios, seed, cfg.probes.val_families
    )
    test = [e for e in pool if e.split == "test"]
    deceptive = [e for e in test if e.condition == "deceptive"]
    spontaneous = [e for e in test if e.condition == "spontaneous"]
    rng.shuffle(deceptive)
    rng.shuffle(spontaneous)
    qa = [
        e
        for e in facts.instructed_qa(10_000, cfg.seed)
        if e.split == "test" and e.condition == "honest"
    ]
    rng.shuffle(qa)
    n = cfg.steering.eval_scenarios
    return deceptive[:n], spontaneous[:n], qa[: cfg.steering.eval_questions]


Steered = dict[str, list[Example]]


def generate_steered(cfg: Config, lm: LanguageModel, bundle: ProbeBundle) -> dict[float, Steered]:
    """Replies to every evaluation prompt at every alpha (greedy, so runs are reproducible)."""
    deceptive, spontaneous, qa = eval_prompts(cfg)
    out: dict[float, Steered] = {}
    for alpha in cfg.steering.alphas:
        edits = truth_serum(bundle, alpha)
        groups: Steered = {}
        for name, group, max_new in (
            ("deceptive", deceptive, cfg.steering.max_new_tokens),
            ("spontaneous", spontaneous, cfg.steering.max_new_tokens),
            ("qa", qa, 48),
        ):
            replies = generate_replies(
                lm,
                [e.messages for e in group],
                max_new_tokens=max_new,
                batch_size=cfg.data.batch_size,
                temperature=0.0,
                top_p=1.0,
                seed=cfg.seed,
                edits=edits,
            )
            groups[name] = [
                e.model_copy(update={"response": r}) for e, r in zip(group, replies, strict=True)
            ]
        out[alpha] = groups
        log.info("generated alpha %.2f", alpha)
    return out


def judge_steered(
    cfg: Config, judge: LanguageModel, steered: dict[float, Steered], layers: list[int]
) -> dict[str, Any]:
    """Judge every steered reply with the unsteered judge and write ``steering.json``."""
    rows: list[dict[str, Any]] = []
    samples: dict[str, list[dict[str, str]]] = {}
    for alpha, groups in steered.items():
        row: dict[str, Any] = {"alpha": alpha}
        for name in ("deceptive", "spontaneous"):
            labels = [v.label for v in judge_examples(judge, groups[name], cfg.data.batch_size)]
            row[name] = {
                "n": len(labels),
                "deceptive_rate": sum(lab == 1 for lab in labels) / len(labels),
                "honest_rate": sum(lab == 0 for lab in labels) / len(labels),
                "unclear_rate": sum(lab is None for lab in labels) / len(labels),
            }
            samples.setdefault(name, []).extend(
                {
                    "alpha": str(alpha),
                    "id": e.id,
                    "question": e.question or "",
                    "reply": e.response or "",
                }
                for e in groups[name][:3]
            )
        qa = groups["qa"]
        correct = [facts.contains_answer(e.response or "", e.answer_key or []) for e in qa]
        row["qa_accuracy"] = sum(correct) / len(correct)
        rows.append(row)
        log.info("alpha %.2f: %s", alpha, row)
    result = {
        "model": cfg.model.name,
        "judge": judge.name,
        "layers": layers,
        "rows": rows,
        "samples": samples,
    }
    cfg.probes.out_dir.mkdir(parents=True, exist_ok=True)
    (cfg.probes.out_dir / "steering.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
