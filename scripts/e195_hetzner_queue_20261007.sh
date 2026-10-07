#!/bin/bash
set -euo pipefail
cd /root/prc-2026
for fold in 5 6; do
  other=$((fold+6))
  taskset -c 0,2,4,6,8,10 nice -n 10 .venv/bin/python -u data/e195_stage_inputs_20261007/scripts/e195_current_chain.py fit --input data/e195_stage_inputs_20261007 --output "data/e195_stage_models_20261007/fold_${fold}_${other}" --host hetzner --fold "$fold" --registration-commit 9078e3508e81bb7c1e6cee432226063b0067b44e
done
