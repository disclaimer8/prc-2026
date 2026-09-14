#!/bin/zsh
# Wait for the daily limit reset (00:05 UTC), then upload the given versions one by one.
set -uo pipefail
cd "${0:a:h}/.."
target=$(date -u -j -f "%Y-%m-%d %H:%M" "$(date -u -v+1d +%Y-%m-%d) 00:05" +%s)
while [ "$(date -u +%s)" -lt "$target" ]; do sleep 60; done
for v in "$@"; do echo "[$(date -u +%H:%M)Z] upload v$v"; uv run python src/upload.py "$v"; sleep 60; done
