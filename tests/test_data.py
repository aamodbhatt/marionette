"""Dataset generators: labels correct by construction, disjoint splits, working verification."""

from __future__ import annotations

import re

import pytest

from marionette.data import controls, facts, scenarios
from marionette.data.judge import Verdict, apply_verdict
from marionette.data.schema import Example, Message, read_jsonl, write_jsonl

VAL = ["landlord", "insurance_claim", "pet_breeder"]


def test_arithmetic_and_comparison_statements_are_labelled_correctly() -> None:
    for ex in facts.fact_statements(10_000, seed=0):
        m = re.fullmatch(r"(\d+) ([+x]) (\d+) = (\d+)\.", ex.response or "")
        if m:
            a, op, b, value = int(m[1]), m[2], int(m[3]), int(m[4])
            truth = a * b if op == "x" else a + b
            assert (value == truth) == (ex.label == 0), ex.id
        m = re.fullmatch(r"(\d+) is larger than (\d+)\.", ex.response or "")
        if m:
            assert (int(m[1]) > int(m[2])) == (ex.label == 0), ex.id


def test_table_facts_true_statement_uses_the_table_and_false_one_does_not() -> None:
    for ex in facts.fact_statements(10_000, seed=0):
        if ex.family != "capital":
            continue
        country = re.match(r"The capital of (.+) is (.+)\.", ex.response or "")
        assert country is not None
        is_true = facts.CAPITALS[country[1]] == country[2]
        assert is_true == (ex.label == 0)


def test_fact_splits_are_by_fact_so_true_and_false_versions_share_a_split() -> None:
    by_key: dict[str, set[str]] = {}
    for ex in facts.fact_statements(10_000, seed=0):
        by_key.setdefault(ex.id.rsplit("-", 1)[0], set()).add(ex.split)
    assert all(len(s) == 1 for s in by_key.values())


@pytest.mark.parametrize(
    ("label", "reply", "ok"),
    [
        (0, "The capital of France is Paris.", True),
        (0, "It is Lyon.", False),
        (1, "The capital of France is Marseille.", True),
        (1, "Paris, of course.", False),
    ],
)
def test_instructed_verification(label: int, reply: str, ok: bool) -> None:
    ex = Example(
        id="x",
        dataset="instructed",
        family="capital",
        split="train",
        condition="deceptive" if label else "honest",
        label=label,
        messages=[Message(role="user", content="q")],
        response=reply,
        answer_key=[facts._word("Paris")],
    )
    assert (facts.verify_answer(ex) is None) == ok


def test_element_symbols_match_case_sensitively() -> None:
    key = [facts._word("Co", case_sensitive=True)]
    assert facts.contains_answer("The symbol is Co.", key)
    assert not facts.contains_answer("The symbol is CO2.", key)
    assert not facts.contains_answer("It's cobalt", key)


def test_family_splits_are_disjoint_and_validation_comes_from_training_families() -> None:
    exs = scenarios.scenario_examples(3, 2, seed=0, val_families=VAL)
    fams = {s: {e.family for e in exs if e.split == s} for s in ("train", "val", "test")}
    assert fams["val"] == set(VAL)
    assert not (fams["train"] & fams["test"]) and not (fams["val"] & fams["test"])
    assert all(scenarios.family(f).split == "train" for f in fams["val"])
    assert len(fams["test"]) >= 5


def test_minimal_pairs_differ_only_in_the_decision_sentence() -> None:
    exs = scenarios.scenario_examples(4, 0, seed=1, val_families=VAL)
    by_id = {e.id: e for e in exs}
    for e in exs:
        if e.condition != "deceptive":
            continue
        twin = by_id[e.id.replace("-deceptive", "-honest")]
        assert e.label == 1 and twin.label == 0
        assert e.messages[1] == twin.messages[1]
        a, b = e.messages[0].content, twin.messages[0].content
        assert a != b
        assert _strip_decision(a) == _strip_decision(b)


def _strip_decision(prompt: str) -> str:
    fams = scenarios.FAMILIES
    pool = (
        *scenarios.DECEPTIVE_DECISIONS,
        *scenarios.HONEST_DECISIONS,
        *(d for f in fams for d in (*f.deceptive_decisions, *f.honest_decisions)),
    )
    hits = [d for d in pool if d in prompt]
    assert len(hits) == 1, hits
    return prompt.replace(hits[0], "")


def test_spontaneous_examples_have_no_label_and_no_scripted_decision() -> None:
    exs = scenarios.scenario_examples(1, 3, seed=0, val_families=VAL)
    spont = [e for e in exs if e.condition == "spontaneous"]
    assert spont and all(e.label is None for e in spont)
    decisions = scenarios.DECEPTIVE_DECISIONS + scenarios.HONEST_DECISIONS
    assert not any(d in e.messages[0].content for e in spont for d in decisions)


def test_generator_is_deterministic_and_templates_are_fully_filled() -> None:
    a = scenarios.scenario_examples(2, 1, seed=7, val_families=VAL)
    b = scenarios.scenario_examples(2, 1, seed=7, val_families=VAL)
    assert [e.model_dump() for e in a] == [e.model_dump() for e in b]
    for e in a:
        assert "{" not in e.messages[0].content and "}" not in e.messages[0].content


def test_unknown_validation_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown validation families"):
        scenarios.scenario_examples(1, 0, seed=0, val_families=["nope"])


def test_controls_are_honest_and_benign_split_in_half() -> None:
    ctrl = controls.control_examples(6, seed=0)
    assert {e.dataset for e in ctrl} == {"confession", "fiction", "persona"}
    assert all(e.label == 0 for e in ctrl)
    rows = [f"Benign prompt number {i} about cooking pasta?" for i in range(40)]
    benign = controls.benign_examples(rows, 20, seed=0)
    assert sum(e.split == "val" for e in benign) == 10


def _roleplay(label: int | None) -> Example:
    return Example(
        id="r",
        dataset="roleplay",
        family="used_car",
        split="train",
        condition="deceptive",
        label=label,
        messages=[Message(role="user", content="q")],
        response="reply",
        truth="t",
        question="q",
    )


def test_apply_verdict_keeps_agreeing_constructed_labels_and_labels_spontaneous() -> None:
    agree = apply_verdict(_roleplay(1), Verdict(0.05, 0.9, 0.05), 0.7)
    assert agree.kept and agree.label == 1
    disagree = apply_verdict(_roleplay(1), Verdict(0.9, 0.05, 0.05), 0.7)
    assert not disagree.kept and disagree.drop_reason == "reply did not follow its condition"
    unsure = apply_verdict(_roleplay(1), Verdict(0.4, 0.5, 0.1), 0.7)
    assert not unsure.kept
    spont = apply_verdict(_roleplay(None), Verdict(0.85, 0.1, 0.05), 0.7)
    assert spont.kept and spont.label == 0


def test_jsonl_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    exs = scenarios.scenario_examples(1, 1, seed=0, val_families=VAL)
    path = tmp_path / "x.jsonl"
    assert write_jsonl(path, exs) == len(exs)
    assert [e.model_dump() for e in read_jsonl(path)] == [e.model_dump() for e in exs]


def test_steering_eval_uses_fresh_prompts_from_test_families_only() -> None:
    from marionette.config import load_config
    from marionette.steer_eval import eval_prompts

    cfg = load_config("configs/smoke.toml")
    deceptive, spontaneous, qa = eval_prompts(cfg)
    assert deceptive and spontaneous and qa
    assert {scenarios.family(e.family).split for e in deceptive + spontaneous} == {"test"}
    assert all(e.condition == "deceptive" for e in deceptive)
    assert all(e.split == "test" and e.condition == "honest" for e in qa)
    trained = scenarios.scenario_examples(
        cfg.data.scenarios_per_family, 0, cfg.seed, cfg.probes.val_families
    )
    seen = {e.messages[0].content + e.messages[1].content for e in trained}
    fresh = [e for e in deceptive if e.messages[0].content + e.messages[1].content not in seen]
    assert len(fresh) >= len(deceptive) // 2
