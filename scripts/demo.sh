#!/usr/bin/env bash
# Serve the app on the GPU, record the README GIF and screenshots, then stop the server.
#   bash scripts/demo.sh configs/qwen3-4b.toml   (on a shared GPU, inside one queue slot)
# The demo's selection rule needs the truth-serum dose-response, so it is computed first if
# the profile has none yet.
set -euo pipefail
cfg="${1:-configs/qwen3-4b.toml}"
results="$(uv run python -c "from marionette.config import load_config; print(load_config('$cfg').probes.out_dir)")"
[ -f "$results/steering.json" ] || uv run marionette steer-eval -c "$cfg"
uv run marionette serve -c "$cfg" &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  curl -sf http://127.0.0.1:8096/api/info > /dev/null && break
  kill -0 "$server" 2>/dev/null || { echo "server exited during startup" >&2; exit 1; }
  sleep 2
done
uv run python scripts/capture_demo.py --url http://127.0.0.1:8096 --results "$results"
