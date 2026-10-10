#!/usr/bin/env bash
# Journal 453: independent replay of the final assembly (journal 436/438) on prc-cloud.
# Payload of the transient systemd unit prc-final-replay-20261010; launched once, no retry, no --refit.
#   systemd-run --unit=prc-final-replay-20261010 -p MemoryMax=24G -p MemorySwapMax=0 -p CPUQuota=800% \
#     -p AllowedCPUs=0-7 -p Nice=10 -p RuntimeMaxSec=3600 -p Restart=no \
#     -p StandardOutput=append:/root/prc-final-logs/replay_20261010.log -p StandardError=append:/root/prc-final-logs/replay_20261010.log \
#     /bin/bash /root/prc-final-logs/run_replay_20261010.sh
# Same adapter (scripts/final_tail_assembly_20261008.py, SHA 9ceab78e...) and contract
# (data/final_tail_20261009/contract.json, SHA 6af6c4ca...) as the Mac build; fresh root and workspace.
set -euo pipefail
R=/root/prc-final-replay-20261010
PY=/root/prc-2026/.venv/bin/python
W=$R/data/final_tail_20261009/work_prc_cloud_20261010
cd "$R"
echo "START $(date -u +%FT%T.%NZ) host=$(hostname) arch=$(uname -m)"
sha256sum scripts/final_tail_assembly_20261008.py data/final_tail_20261009/contract.json data/today_chain_20261006/setup.json
"$PY" -c 'import sys; print("python", sys.version)'
"$PY" -u scripts/final_tail_assembly_20261008.py prepare --contract data/final_tail_20261009/contract.json --work "$W"
"$PY" -u scripts/final_tail_assembly_20261008.py assemble --work "$W" --repeat-e156
echo "END $(date -u +%FT%T.%NZ)"
