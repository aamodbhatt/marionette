"""Every shipped profile parses and validates."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from marionette.config import Config, load_config

CONFIGS = sorted(Path(__file__).resolve().parents[1].joinpath("configs").glob("*.toml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_profiles_load(path: Path) -> None:
    cfg = load_config(path)
    assert cfg.probes.val_families


def test_unknown_keys_fail_loudly() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"run_name": "x", "probes": {"val_families": ["a"]}, "typo": 1})


def test_sae_buffer_must_hold_two_batches() -> None:
    with pytest.raises(ValidationError, match="two SAE batches"):
        Config.model_validate(
            {
                "run_name": "x",
                "probes": {"val_families": ["a"]},
                "sae": {"buffer_tokens": 100, "batch_tokens": 64},
            }
        )
