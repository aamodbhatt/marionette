"""Command-line entry point: one sub-command per pipeline stage, all driven by one TOML profile."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any

import typer

from marionette.config import Config, load_config

app = typer.Typer(
    add_completion=False, no_args_is_help=True, help="marionette: a polygraph for LLMs."
)
sae_app = typer.Typer(no_args_is_help=True, help="Sparse autoencoder stages.")
app.add_typer(sae_app, name="sae")

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="TOML profile.")]


def _setup(config: Path) -> Config:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    from marionette.runtime import configure_matmul, seed_everything

    cfg = load_config(config)
    seed_everything(cfg.seed)
    configure_matmul()
    return cfg


def _load_lm(cfg: Config, name: str | None = None, n_layers: int | None = None):  # type: ignore[no-untyped-def]
    from marionette.lm import load_language_model
    from marionette.runtime import resolve_device

    return load_language_model(
        name or cfg.model.name,
        cfg.model.dtype,
        resolve_device(cfg.device),
        cfg.model.revision,
        n_layers=n_layers,
    )


@app.command()
def data(config: ConfigOpt) -> None:
    """Build all deception datasets: prompts, on-policy replies, verification."""
    cfg = _setup(config)
    from marionette.data.build import build_specs, generate, verify_and_write
    from marionette.data.controls import load_benign_rows

    examples = build_specs(cfg, load_benign_rows(cfg.data.benign_repo, cfg.data.benign_pattern))
    holder = [_load_lm(cfg)]
    generate(cfg, holder[0], examples)
    report = verify_and_write(cfg, _swap_to_judge(cfg, holder), examples)
    typer.echo(json.dumps(report, indent=2))


def _swap_to_judge(cfg: Config, holder: list[Any]) -> Any:
    """The judge model. A separate judge is loaded only after the generator is freed.

    The generator arrives in a one-element list that this function empties, so no caller
    frame keeps a reference that would pin its weights on the GPU next to the judge's.
    """
    if cfg.data.judge_model is None:
        return holder[0]
    import gc

    import torch

    holder.clear()
    gc.collect()
    torch.cuda.empty_cache()
    return _load_lm(cfg, cfg.data.judge_model)


@app.command()
def capture(config: ConfigOpt) -> None:
    """Pool response activations at every layer for every kept example."""
    cfg = _setup(config)
    from marionette.capture import capture as run_capture
    from marionette.data.build import examples_path
    from marionette.data.schema import read_jsonl

    lm = _load_lm(cfg)
    run_capture(
        lm,
        list(read_jsonl(examples_path(cfg))),
        cfg.data.dir / "acts",
        cfg.data.capture_batch,
        cfg.data.max_seq_len,
    )


@app.command()
def probes(config: ConfigOpt) -> None:
    """Layer sweep, generalisation matrix, calibration, FPR; exports the probe weights."""
    cfg = _setup(config)
    from marionette.capture import load_activations
    from marionette.evaluation import run

    results = run(load_activations(cfg.data.dir / "acts"), cfg)
    # The dataset report travels with the probe results it describes.
    report = (cfg.data.dir / "build_report.json").read_text()
    (cfg.probes.out_dir / "dataset.json").write_text(report)
    typer.echo(
        json.dumps(
            {
                "choice": results["choice"],
                "auroc": results["final"]["logistic"]["auroc"],
                "recall": results["final"]["logistic"]["recall_at_benign_fpr"],
                "fpr": results["final"]["logistic"]["fpr_at_threshold"],
            },
            indent=2,
        )
    )


@app.command("steer-eval")
def steer_eval(config: ConfigOpt) -> None:
    """Deception rate and QA accuracy versus truth-serum strength."""
    cfg = _setup(config)
    from marionette.steer_eval import generate_steered, judge_steered
    from marionette.steering import ProbeBundle

    bundle = ProbeBundle.load(cfg.probes.weights_dir)
    holder = [_load_lm(cfg)]
    steered = generate_steered(cfg, holder[0], bundle)
    result = judge_steered(cfg, _swap_to_judge(cfg, holder), steered, sorted(bundle.steer))
    typer.echo(json.dumps(result["rows"], indent=2))


def _probe_layer(cfg: Config) -> int:
    from marionette.steering import ProbeBundle

    return ProbeBundle.load(cfg.probes.weights_dir).layer


@sae_app.command("tokens")
def sae_tokens(config: ConfigOpt) -> None:
    """Tokenise the SAE training mixture (CPU and network only)."""
    cfg = _setup(config)
    from transformers import AutoTokenizer

    from marionette.sae_train import prepare_tokens

    prepare_tokens(cfg, AutoTokenizer.from_pretrained(cfg.model.name, revision=cfg.model.revision))


@sae_app.command("train")
def sae_train(
    config: ConfigOpt,
    layer: Annotated[int | None, typer.Option(help="Default: the probe layer.")] = None,
) -> None:
    """Train the BatchTopK SAE on activations streamed from the truncated model."""
    cfg = _setup(config)
    from marionette.sae_train import train

    layer = _probe_layer(cfg) if layer is None else layer
    stats = train(cfg, _load_lm(cfg, n_layers=layer + 1), layer)
    typer.echo(json.dumps({k: v for k, v in stats.items() if k != "history"}, indent=2))


@sae_app.command("eval")
def sae_eval(config: ConfigOpt) -> None:
    """FVU, L0, dead fraction and CE loss recovered on held-out tokens."""
    cfg = _setup(config)
    from marionette.runtime import resolve_device
    from marionette.sae import BatchTopKSAE
    from marionette.sae_train import evaluate, write_eval

    sae = BatchTopKSAE.load(cfg.sae.dir, resolve_device(cfg.device))
    stats = evaluate(cfg, _load_lm(cfg), sae)
    write_eval(cfg, stats)
    typer.echo(json.dumps(stats, indent=2))


@app.command()
def features(config: ConfigOpt) -> None:
    """Deception alignment, top activations and auto-labels for the most relevant latents."""
    cfg = _setup(config)
    from marionette.data.build import examples_path
    from marionette.data.schema import read_jsonl
    from marionette.features import run
    from marionette.runtime import resolve_device
    from marionette.sae import BatchTopKSAE
    from marionette.steering import ProbeBundle

    sae = BatchTopKSAE.load(cfg.sae.dir, resolve_device(cfg.device))
    result = run(
        cfg,
        _load_lm(cfg),  # full model: auto-labelling generates text
        sae,
        ProbeBundle.load(cfg.probes.weights_dir),
        list(read_jsonl(examples_path(cfg))),
    )
    typer.echo(json.dumps(result["summary"], indent=2))


@app.command()
def serve(
    config: ConfigOpt, port: Annotated[int | None, typer.Option(help="Default: serve.port")] = None
) -> None:
    """Run the polygraph web app (needs the probe weights; the SAE is optional)."""
    cfg = _setup(config)
    import uvicorn

    from marionette.runtime import resolve_device
    from marionette.sae import BatchTopKSAE
    from marionette.server.app import create_app
    from marionette.server.engine import LiveEngine, load_feature_infos
    from marionette.steering import ProbeBundle

    bundle = ProbeBundle.load(cfg.probes.weights_dir)
    lm = _load_lm(cfg)
    sae = None
    # Both files must be present: a directory with only the metadata (an interrupted copy, say)
    # should disable the thought stream, not crash the server.
    if all((cfg.sae.dir / name).is_file() for name in ("sae.json", "sae.safetensors")):
        sae = BatchTopKSAE.load(cfg.sae.dir, resolve_device(cfg.device))
    else:
        logging.getLogger(__name__).warning(
            "no complete SAE (sae.json + sae.safetensors) at %s: thought stream disabled",
            cfg.sae.dir,
        )
    infos = load_feature_infos(cfg.probes.weights_dir / "features.json") if sae else []
    engine = LiveEngine(lm, bundle, sae, infos, cfg.serve.max_alpha, profile=cfg.run_name)
    web = create_app(engine, cfg.probes.out_dir, cfg.probes.weights_dir, infos)
    uvicorn.run(web, host=cfg.serve.host, port=port or cfg.serve.port, log_level="warning")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
