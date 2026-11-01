"""HTTP layer: a FastAPI app over any ``Engine``, plus the static single-page UI.

``POST /api/chat`` streams newline-delimited JSON events (``start``, ``token``..., ``done`` or
``error``). Generation is synchronous and runs in Starlette's threadpool; the engine serialises
requests with its own lock, so one GPU model serves the page safely.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from marionette.server.engine import ChatRequest, Engine, FeatureInfo
from marionette.server.presets import PRESETS

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def research_summary(results_dir: Path, weights_dir: Path) -> dict[str, Any]:
    """The dashboard payload: probe study, steering curve, SAE stats, top deception features."""
    out: dict[str, Any] = {}
    probes = results_dir / "probes.json"
    if probes.is_file():
        p = json.loads(probes.read_text())
        final = {
            k: {m: v for m, v in metrics.items() if m != "scores"}
            for k, metrics in p["final"].items()
        }
        out["probes"] = {
            "choice": p["choice"],
            "sweep": p["sweep"],
            "counts": p["counts"],
            "generalisation": p["generalisation"],
            "final": final,
        }
    for name in ("steering", "sae", "dataset"):
        path = results_dir / f"{name}.json"
        if path.is_file():
            data = json.loads(path.read_text())
            data.pop("samples", None)
            out[name] = data
    feats = weights_dir / "features.json"
    if feats.is_file():
        f = json.loads(feats.read_text())
        ranked = sorted(f["features"], key=lambda x: -x["deception_contribution"])
        out["features"] = {
            "summary": f["summary"],
            "top_deception": [
                {
                    k: x[k]
                    for k in (
                        "id",
                        "label",
                        "deception_contribution",
                        "deception_auroc",
                        "detection_auroc",
                        "cosine_with_serum",
                    )
                }
                for x in ranked[:15]
            ],
        }
    return out


def create_app(
    engine: Engine, results_dir: Path, weights_dir: Path, features: list[FeatureInfo] | None = None
) -> FastAPI:
    app = FastAPI(title="marionette", docs_url="/api/docs", redoc_url=None)
    labelled = features or []
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/info")
    def info() -> dict[str, Any]:
        return engine.info()

    @app.get("/api/presets")
    def presets() -> list[dict[str, Any]]:
        return PRESETS

    @app.get("/api/features")
    def feature_search(
        q: str = Query("", max_length=80), limit: int = Query(30, ge=1, le=200)
    ) -> list[dict[str, Any]]:
        needle = q.lower().strip()
        hits = [f for f in labelled if needle in f.label.lower() or needle == str(f.id)]
        return [
            {"id": f.id, "label": f.label, "deception_contribution": f.deception_contribution}
            for f in hits[:limit]
        ]

    @app.get("/api/research")
    def research() -> dict[str, Any]:
        return research_summary(results_dir, weights_dir)

    @app.post("/api/chat")
    def chat(request: ChatRequest) -> StreamingResponse:
        if request.messages[-1].role != "user":
            raise HTTPException(422, "the last message must be from the user")

        def events() -> Iterator[bytes]:
            try:
                for event in engine.stream(request):
                    yield (json.dumps(event) + "\n").encode()
            except ValueError as err:
                yield (json.dumps({"type": "error", "message": str(err)}) + "\n").encode()
            except Exception:
                log.exception("generation failed")
                yield (
                    json.dumps({"type": "error", "message": "generation failed; see server log"})
                    + "\n"
                ).encode()

        return StreamingResponse(events(), media_type="application/x-ndjson")

    return app
