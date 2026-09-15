#!/bin/zsh
# Upload a finished submission as soon as its file appears and the daily limit allows; wait for the score. Log: data/submit_15sep.log
#   src/submit_when_ready.sh 26 data/run_final_e28.log
cd "${0:a:h}/.."
v=$1 log=$2
until grep -q "FINAL submissions/gentle-igloo_v$v.parquet" $log 2>/dev/null || grep -qE "ABORT|Traceback|Error" $log 2>/dev/null; do sleep 60; done
grep -qE "ABORT|Traceback|Error" $log && { echo "[$(date -u +%H:%M)Z] v$v build failed, not uploaded"; exit 1; }
.venv/bin/python src/upload.py $v 2>&1 | grep -v Warning | cut -c1-60
echo "[$(date -u +%H:%M)Z] uploaded v$v"
.venv/bin/python - $v <<'PY'
import os, sys, time, json
os.environ['POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR'] = 'load_as_storage'
from dotenv import load_dotenv; from minio import Minio
load_dotenv('.env')
c = Minio("s3.opensky-network.org", access_key=os.environ["PRC_ACCESS_KEY"], secret_key=os.environ["PRC_SECRET_KEY"], secure=True)
name = f"gentle-igloo_v{sys.argv[1]}.parquet_result.json"
for _ in range(90):
    try:
        r = json.loads(c.get_object("prc-2026-gentle-igloo", name).read()); print("SCORE", sys.argv[1], r.get("status"), r.get("score")); break
    except Exception: time.sleep(30)
else: print("timeout")
PY
