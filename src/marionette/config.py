"""Typed, validated configuration for every pipeline stage.

One TOML file describes a whole profile (model, datasets, probes, steering, SAE, serving).
Every stage reads the same file, so a run is reproducible from its config plus the seed.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

DType = Literal["bfloat16", "float32"]


class _Strict(BaseModel):
    """Base model that rejects unknown keys so typos in TOML fail loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(_Strict):
    """The chat model under study (always run in non-thinking mode)."""

    name: str = "Qwen/Qwen3-4B"
    revision: str | None = None
    dtype: DType = "bfloat16"


class DataConfig(_Strict):
    """Deception datasets: sizes, generation and judging."""

    dir: Path = Path("data/qwen3-4b")
    judge_model: str | None = Field(
        None, description="Judge model id; None means the studied model judges its own responses."
    )
    scenarios_per_family: int = Field(24, ge=1, description="Instances per family and condition.")
    spontaneous_per_family: int = Field(24, ge=0)
    instructed_questions: int = Field(240, ge=2)
    fact_statements: int = Field(600, ge=2)
    controls_per_kind: int = Field(80, ge=1)
    benign_prompts: int = Field(240, ge=2)
    benign_repo: str = "HuggingFaceH4/ultrachat_200k"
    benign_pattern: str = "data/test_sft-*.parquet"
    max_new_tokens: int = Field(96, ge=8)
    batch_size: int = Field(48, ge=1)
    temperature: float = Field(0.7, ge=0)
    top_p: float = Field(0.9, gt=0, le=1)
    judge_min_confidence: float = Field(0.7, ge=0.34, le=1)
    capture_batch: int = Field(16, ge=1)
    max_seq_len: int = Field(768, ge=64)


class ProbeConfig(_Strict):
    """Linear probes: which families validate the layer choice, and the L2 grid."""

    val_families: list[str] = Field(min_length=1)
    l2_grid: list[float] = Field(default_factory=lambda: [1e-3, 1e-2, 1e-1, 1.0])
    max_iter: int = Field(300, ge=10)
    benign_fpr: float = Field(0.01, gt=0, lt=0.5)
    out_dir: Path = Path("results/qwen3-4b")
    weights_dir: Path = Path("weights/qwen3-4b")


class SteeringConfig(_Strict):
    """Truth serum: subtract the deception direction during generation."""

    band: int = Field(
        4, ge=1, description="Steer this many consecutive layers starting at the probe layer."
    )
    alphas: list[float] = Field(default_factory=lambda: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
    eval_scenarios: int = Field(96, ge=1)
    eval_questions: int = Field(96, ge=1)
    max_new_tokens: int = Field(80, ge=8)


class SAEConfig(_Strict):
    """BatchTopK sparse autoencoder with an AuxK dead-latent loss."""

    n_latents: int = Field(16384, ge=2)
    k: int = Field(64, ge=1)
    aux_k: int = Field(512, ge=1)
    aux_coef: float = Field(1.0 / 32.0, ge=0)
    dead_tokens: int = Field(2_000_000, ge=1)
    lr: float = Field(2e-4, gt=0)
    warmup_steps: int = Field(500, ge=0)
    decay_start: float = Field(0.8, gt=0, le=1)
    threshold_beta: float = Field(0.999, gt=0, lt=1)
    batch_tokens: int = Field(4096, ge=8)
    train_tokens: int = Field(20_000_000, ge=1)
    heldout_tokens: int = Field(1_000_000, ge=1)
    ctx_len: int = Field(256, ge=16)
    lm_batch_contexts: int = Field(32, ge=1)
    buffer_tokens: int = Field(262_144, ge=16)
    fineweb_repo: str = "HuggingFaceFW/fineweb-edu"
    fineweb_files: list[str] = Field(default_factory=lambda: ["sample/10BT/000_00000.parquet"])
    chat_repo: str = "HuggingFaceH4/ultrachat_200k"
    chat_files: list[str] = Field(default_factory=lambda: ["data/train_sft-*.parquet"])
    chat_fraction: float = Field(0.5, ge=0, le=1)
    log_every: int = Field(100, ge=1)
    seed: int = 0
    dir: Path = Path("artifacts/qwen3-4b/sae")

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.k > self.n_latents:
            raise ValueError(f"k={self.k} exceeds n_latents={self.n_latents}")
        if self.buffer_tokens < 2 * self.batch_tokens:
            raise ValueError("buffer_tokens must hold at least two SAE batches")
        return self


class AutoInterpConfig(_Strict):
    """Labelling the most deception-relevant and most frequent features with the model itself."""

    n_features: int = Field(300, ge=1)
    n_deception: int = Field(
        120, ge=0, description="Of those, how many are picked by deception relevance."
    )
    collect_contexts: int = Field(4096, ge=8)
    top_examples: int = Field(16, ge=4)
    label_examples: int = Field(10, ge=2)
    window_before: int = Field(24, ge=1)
    window_after: int = Field(4, ge=0)
    batch_size: int = Field(16, ge=1)
    max_label_tokens: int = Field(16, ge=4)


class ServeConfig(_Strict):
    """Web app settings."""

    host: str = "127.0.0.1"
    port: int = 8096
    max_new_tokens: int = Field(160, ge=1, le=512)
    max_alpha: float = Field(4.0, gt=0)


class Config(_Strict):
    """A complete marionette profile."""

    run_name: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    seed: int = 0
    device: str = "auto"
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    probes: ProbeConfig
    steering: SteeringConfig = SteeringConfig()
    sae: SAEConfig = SAEConfig()
    autointerp: AutoInterpConfig = AutoInterpConfig()
    serve: ServeConfig = ServeConfig()


def load_config(path: Path | str) -> Config:
    """Parse and validate a TOML profile."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("rb") as f:
        raw = tomllib.load(f)
    return Config.model_validate(raw)
