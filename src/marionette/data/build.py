"""Build every dataset end to end: prompts -> on-policy replies -> verification -> JSONL.

Output (``<data.dir>/``): ``examples.jsonl`` (every example, kept or dropped, with the reason)
and ``build_report.json`` (counts, drop rates and how often the judge agrees with answer keys).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from marionette.config import Config
from marionette.data import controls, facts, scenarios
from marionette.data.judge import CHOICES, Verdict, apply_verdict, judge_messages
from marionette.data.schema import Example, write_jsonl
from marionette.generation import choice_probabilities, generate_replies
from marionette.lm import LanguageModel

log = logging.getLogger(__name__)


def examples_path(cfg: Config) -> Path:
    return cfg.data.dir / "examples.jsonl"


def build_specs(cfg: Config, benign_rows: list[Any]) -> list[Example]:
    """All examples before generation (deterministic given the config's seed)."""
    d, seed = cfg.data, cfg.seed
    return [
        *facts.fact_statements(d.fact_statements, seed),
        *facts.instructed_qa(d.instructed_questions, seed),
        *scenarios.scenario_examples(
            d.scenarios_per_family, d.spontaneous_per_family, seed, cfg.probes.val_families
        ),
        *controls.control_examples(d.controls_per_kind, seed),
        *controls.benign_examples(benign_rows, d.benign_prompts, seed),
    ]


def judge_examples(judge: LanguageModel, examples: list[Example], batch_size: int) -> list[Verdict]:
    probs = choice_probabilities(judge, [judge_messages(e) for e in examples], CHOICES, batch_size)
    return [Verdict(*(float(p) for p in row)) for row in probs]


def generate(cfg: Config, lm: LanguageModel, examples: list[Example]) -> None:
    """Fill in on-policy replies for every example that needs one (in place)."""
    todo = [e for e in examples if e.needs_generation]
    log.info("%d examples, %d need generated replies", len(examples), len(todo))
    replies = generate_replies(
        lm,
        [e.messages for e in todo],
        max_new_tokens=cfg.data.max_new_tokens,
        batch_size=cfg.data.batch_size,
        temperature=cfg.data.temperature,
        top_p=cfg.data.top_p,
        seed=cfg.seed,
    )
    for ex, reply in zip(todo, replies, strict=True):
        ex.response = reply
        if not reply:
            ex.kept, ex.drop_reason = False, "empty reply"


def verify_and_write(cfg: Config, judge: LanguageModel, examples: list[Example]) -> dict[str, Any]:
    """Verify labels (answer keys, judge), then write the examples and the build report."""
    # Instructed QA: the answer key decides; the judge also runs so we can measure its accuracy.
    instructed = [e for e in examples if e.dataset == "instructed" and e.kept]
    judged_roleplay = [e for e in examples if e.dataset in ("roleplay", "spontaneous") and e.kept]
    verdicts = judge_examples(judge, instructed + judged_roleplay, cfg.data.batch_size)
    agreement: Counter[str] = Counter()
    for ex, verdict in zip(instructed, verdicts[: len(instructed)], strict=True):
        ex.judge = verdict.as_dict()
        reason = facts.verify_answer(ex)
        if reason is not None:
            ex.kept, ex.drop_reason = False, reason
            continue
        agreement["n"] += 1
        agreement["agree"] += int(verdict.label == ex.label)
    for ex, verdict in zip(judged_roleplay, verdicts[len(instructed) :], strict=True):
        apply_verdict(ex, verdict, cfg.data.judge_min_confidence)

    n = write_jsonl(examples_path(cfg), examples)
    report = build_report(examples, agreement)
    (cfg.data.dir / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    log.info("wrote %d examples to %s", n, examples_path(cfg))
    return report


def build_report(examples: list[Example], agreement: Counter[str]) -> dict[str, Any]:
    by_ds: dict[str, dict[str, int]] = {}
    for ex in examples:
        row = by_ds.setdefault(ex.dataset, {"total": 0, "kept": 0, "deceptive": 0, "honest": 0})
        row["total"] += 1
        if ex.kept and ex.label is not None:
            row["kept"] += 1
            row["deceptive" if ex.label == 1 else "honest"] += 1
    drops = Counter(f"{e.dataset}: {e.drop_reason}" for e in examples if not e.kept)
    spont = [e for e in examples if e.dataset == "spontaneous" and e.kept]
    return {
        "datasets": by_ds,
        "drop_reasons": dict(drops.most_common()),
        "judge_vs_answer_key": {
            "n": agreement["n"],
            "accuracy": agreement["agree"] / agreement["n"] if agreement["n"] else None,
        },
        "spontaneous_lie_rate": (sum(e.label == 1 for e in spont) / len(spont)) if spont else None,
        "kept_total": sum(e.kept and e.label is not None for e in examples),
    }
