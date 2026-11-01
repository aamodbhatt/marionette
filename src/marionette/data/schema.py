"""The single record type every dataset produces, and its JSONL serialisation.

Label convention used throughout the project: ``1`` means deceptive (or, for plain factual
statements, false) and ``0`` means honest (true). ``None`` means "not decided yet": spontaneous
role-play responses get their label from the judge after generation.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DatasetName = Literal[
    "facts", "instructed", "roleplay", "spontaneous", "confession", "fiction", "persona", "benign"
]
Split = Literal["train", "val", "test"]
Condition = Literal["honest", "deceptive", "spontaneous", "control", "true", "false"]

DECEPTIVE = 1
HONEST = 0

# Datasets that are never trained on: they only measure false positives.
CONTROL_DATASETS: tuple[DatasetName, ...] = ("confession", "fiction", "persona", "benign")


class Message(BaseModel):
    """One chat turn."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class Example(BaseModel):
    """A prompt, the model's response to it, its label and everything needed to verify the label."""

    model_config = ConfigDict(extra="forbid")

    id: str
    dataset: DatasetName
    family: str
    split: Split
    condition: Condition
    label: int | None = Field(None, ge=0, le=1)
    messages: list[Message] = Field(min_length=1)
    response: str | None = None
    truth: str | None = Field(
        None, description="Ground truth the judge compares the reply against."
    )
    question: str | None = None
    answer_key: list[str] | None = Field(None, description="Regexes; a correct answer matches one.")
    judge: dict[str, float] | None = None
    kept: bool = True
    drop_reason: str | None = None

    @property
    def needs_generation(self) -> bool:
        return self.response is None


def write_jsonl(path: Path, examples: Iterable[Example]) -> int:
    """Write examples atomically (temp file + rename); returns the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    n = 0
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")
            n += 1
    tmp.replace(path)
    return n


def read_jsonl(path: Path) -> Iterator[Example]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `marionette data` first")
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield Example.model_validate(json.loads(line))
                except ValueError as err:
                    raise ValueError(f"{path}:{line_no}: invalid example: {err}") from err
