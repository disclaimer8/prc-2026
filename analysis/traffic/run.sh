#!/bin/bash
# usage: run.sh NN_name  (runs scripts/NN_name.py, writes scripts/NN_name.txt)
cd ~/prc-2026/analysis/traffic/scripts || exit 1
POLARS_MAX_THREADS=4 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage uv run --project ~/prc-2026 --with matplotlib python "$1.py" > "$1.txt" 2>&1
echo "exit=$?"
