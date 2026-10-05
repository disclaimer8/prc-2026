#!/bin/bash
# Collect only the complete registered E190 run, then execute its fixed gate.
set -euo pipefail
cd /Users/denyskolomiiets/prc-2026
mkdir data/.e190_collector_20261005.lock
deadline=$((SECONDS + 3600))
while true; do
  status=0
  ssh -o BatchMode=yes -o ConnectTimeout=10 home-pc-lan 'if test -f /home/a1/prc-2026/data/e190_numemb_20261005/complete.json; then exit 0; elif systemctl --user is-active --quiet prc-e190-numemb-20261005.service; then exit 3; else exit 4; fi' || status=$?
  case "$status" in
    0) break ;;
    3) if (( SECONDS >= deadline )); then echo 'Collection deadline reached'; exit 1; fi ;;
    *) echo "Home job or SSH unavailable: status=$status"; exit 1 ;;
  esac
  sleep 30
done
rsync -a home-pc-lan:/home/a1/prc-2026/data/e190_numemb_20261005/ data/e190_numemb_20261005/
.venv/bin/python scripts/e190_numeric_embeddings.py evaluate \
  --root /Users/denyskolomiiets/prc-2026 \
  --bundle /Users/denyskolomiiets/prc-2026/data/e190_inputs_20261005 \
  --out /Users/denyskolomiiets/prc-2026/data/e190_numemb_20261005 \
  --registration-commit 3157b2d7556dec75e7be3472026736411eb45f6a
