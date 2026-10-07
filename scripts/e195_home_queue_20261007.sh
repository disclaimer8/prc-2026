#!/bin/bash
set -euo pipefail
cd /home/a1/prc-2026
.venv/bin/python -u data/e195_stage_inputs_20261007/scripts/e195_current_chain.py fit --input data/e195_stage_inputs_20261007 --output data/e195_stage_models_20261007/fold_3_9 --host home --fold 3 --registration-commit 9078e3508e81bb7c1e6cee432226063b0067b44e
.venv/bin/python -u data/e195_stage_inputs_20261007/scripts/e195_current_chain.py fit --input data/e195_stage_inputs_20261007 --output data/e195_stage_models_20261007/fold_4_10 --host home --fold 4 --registration-commit 9078e3508e81bb7c1e6cee432226063b0067b44e
