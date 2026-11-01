#!/usr/bin/env bash
# The whole study in one go (inside one queue slot on a shared GPU): a tiny smoke run of every
# stage, the real profile, then the README demo recording.
#   uv run marionette sae tokens -c configs/smoke.toml
#   uv run marionette sae tokens -c configs/qwen3-4b.toml
#   bash scripts/run_all.sh
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$here/pipeline.sh" configs/smoke.toml || echo "!!! smoke run had failures; continuing"
bash "$here/pipeline.sh" configs/qwen3-4b.toml || echo "!!! main run had failures"
bash "$here/demo.sh" configs/qwen3-4b.toml
