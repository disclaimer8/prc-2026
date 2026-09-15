#!/bin/zsh
# E23: six outer folds sequentially (each in its own process to free memory), then the summary with the decision.
cd "${0:a:h}/.."
export FEATURES=data/features_final.parquet DROP_PREFIX=day_,tq_,wxr_ POLARS_MAX_THREADS=10
for pair in "1 7" "2 8" "3 9" "4 10" "5 11" "6 12"; do
  set -- ${=pair}
  [[ -f data/e23/fold_$1_$2.npz ]] && continue
  echo "[$(date +%H:%M)] fold $1+$2"
  .venv/bin/python src/e23_sse_checkpoint.py $1 $2 2>&1 | grep -v Warning || echo "FAILED fold $1+$2"
done
.venv/bin/python src/e23_sse_checkpoint.py summary
echo "[$(date +%H:%M)] done"
