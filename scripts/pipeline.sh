#!/usr/bin/env bash
# Every GPU stage of one profile, in order. On a shared GPU, run it inside one queue slot so the
# wait happens once:
#   bash scripts/pipeline.sh configs/qwen3-4b.toml [first-stage]
# Run `marionette sae tokens` first (CPU and network only, no GPU needed).
# data, capture and probes are required; a failure in a later stage is reported, and the
# remaining stages still run, so one bug late in the chain does not waste the earlier GPU work.
set -uo pipefail
cfg="${1:?usage: pipeline.sh <config.toml> [first-stage]}"
from="${2:-data}"
stages=(data capture probes steer-eval sae-train sae-eval features)
required=" data capture probes "
run=0
failed=0
for stage in "${stages[@]}"; do
  [ "$stage" = "$from" ] && run=1
  [ "$run" = 1 ] || continue
  echo "=== $stage ($(date +%H:%M:%S))"
  case "$stage" in
    sae-train) uv run marionette sae train -c "$cfg" ;;
    sae-eval) uv run marionette sae eval -c "$cfg" ;;
    *) uv run marionette "$stage" -c "$cfg" ;;
  esac
  code=$?
  if [ "$code" != 0 ]; then
    echo "!!! stage $stage failed (exit $code)"
    [[ "$required" == *" $stage "* ]] && exit "$code"
    failed=1
  fi
done
echo "=== pipeline done ($(date +%H:%M:%S)), failures: $failed"
exit "$failed"
