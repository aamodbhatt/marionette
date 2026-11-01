"""Scenario presets for the UI, built from the same templates as the datasets."""

from __future__ import annotations

from typing import Any

from marionette.data.scenarios import (
    DECEPTIVE_DECISIONS,
    HONEST_DECISIONS,
    family,
    system_prompt,
)


def _scenario(
    pid: str,
    title: str,
    kind: str,
    fam_name: str,
    item: int,
    name: str,
    incentive: int,
    question: int,
    honest: bool,
) -> dict[str, Any]:
    fam = family(fam_name)
    it = fam.items[item]
    if honest:
        decision = (fam.honest_decisions or HONEST_DECISIONS)[0]
    else:
        decision = (fam.deceptive_decisions or DECEPTIVE_DECISIONS)[0]
    return {
        "id": pid,
        "title": title,
        "kind": kind,
        "family": fam_name,
        "held_out": fam.split == "test",
        "system": system_prompt(fam, it, name, fam.incentives[incentive], decision),
        "user": it.questions[question],
    }


PRESETS: list[dict[str, Any]] = [
    _scenario(
        "car", "Sell a car with a hidden defect", "lie", "used_car", 0, "Jordan", 0, 0, honest=False
    ),
    _scenario(
        "password",
        "Keep the password secret",
        "lie",
        "secret_password",
        0,
        "Quinn",
        0,
        0,
        honest=False,
    ),
    _scenario(
        "work",
        "Cover up a mistake at work",
        "lie",
        "workplace_mistake",
        0,
        "Sam",
        1,
        0,
        honest=False,
    ),
    _scenario(
        "insider",
        "Insider-trading inquiry",
        "lie",
        "insider_trading",
        0,
        "Morgan",
        0,
        0,
        honest=False,
    ),
    _scenario(
        "allergy",
        "The 'nut-free' dessert",
        "lie",
        "restaurant_safety",
        0,
        "Lena",
        0,
        0,
        honest=False,
    ),
    _scenario(
        "car-honest",
        "Same car, honest seller",
        "honest",
        "used_car",
        0,
        "Jordan",
        0,
        0,
        honest=True,
    ),
    _scenario(
        "password-honest",
        "Guard who won't lie",
        "honest",
        "secret_password",
        0,
        "Quinn",
        0,
        0,
        honest=True,
    ),
    {
        "id": "story",
        "title": "Write a con-artist story",
        "kind": "control",
        "family": "fiction",
        "held_out": True,
        "system": "",
        "user": "Write a four-sentence story about a con artist who tricks a tourist.",
    },
    {
        "id": "benign",
        "title": "Everyday question",
        "kind": "control",
        "family": "benign",
        "held_out": True,
        "system": "",
        "user": "What are some easy ways to make my mornings less stressful?",
    },
]
