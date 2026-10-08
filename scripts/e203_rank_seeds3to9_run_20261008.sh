#!/bin/bash
# Home launcher for scripts/e203_rank_seeds3to9_20261008.py (journal №416).
# Waits for the running E212 GPU unit to leave the GPU (never stops it), then runs the
# seeds 3..9 fit once. systemd-run expands $variables in ExecStart, so the loop lives here.
set -u
cd /home/a1/prc-2026 || exit 2
while systemctl --user is-active --quiet prc-e212-seeds3to9-20261008.service; do sleep 30; done
echo "E212 unit inactive at $(date -u +%FT%TZ); starting E203 ranking seeds 3..9"
exec /home/a1/prc-2026/.venv-numemb/bin/python -u \
  /home/a1/prc-2026/data/e203_rank_seeds3to9_20261008/scripts/e203_rank_seeds3to9_20261008.py \
  --root /home/a1/prc-2026
