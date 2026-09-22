#!/bin/zsh
# Продолжение после снятия третьей полосы: ждём двух полос E92 (по числу файлов), добираем фолд 6,
# затем E95 в ДВЕ полосы (по памяти: один фолд ~10 ГБ из 48, три полосы упирались в компрессор).
set -uo pipefail
setopt NULL_GLOB  # zsh: несовпавший шаблон иначе рубит команду целиком
cd /Users/denyskolomiiets/prc-2026
say() { print -r -- "[$(date +%H:%M)] $*" }
tg() { ssh -o BatchMode=yes hetzner "set -a; . /etc/flightfinder/telegram.env; set +a; curl -s -m 20 -X POST \"https://api.telegram.org/bot\$TELEGRAM_BOT_TOKEN/sendMessage\" -d chat_id=90206238 --data-urlencode text=\"$1\" >/dev/null" 2>/dev/null; }
lane() { local folds=$1 out=$2 log=$3; shift 3; env $@ FOLDS=$folds CV_OUT=$out OMP_NUM_THREADS=6 .venv/bin/python src/cv_fold_preds_nested.py >> $log 2>&1 & }
E92=(NO_AOBT=1 PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 FEATURES=data/features_noaobt_e28.parquet DROP_PREFIX=day_,tq_,wxr_,month)
E95=(CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 CLF_SEEDS=5 FEATURES=data/features_e77.parquet DROP_PREFIX=day_,tq_,wxr_,month)
out=data/cv_preds_e92_noaobt_cw
until [[ $(ls $out/fold_*.npz 2>/dev/null | wc -l | tr -d ' ') -ge 5 ]]; do sleep 60; done
until ! pgrep -qf "cv_fold_preds_neste[d]"; do sleep 30; done
say "две полосы E92 готовы, добираю фолд 6"
lane 6 $out data/e92.log $E92
wait
n=$(ls $out/fold_*.npz | wc -l | tr -d ' '); say "E92 собрано $n из 6"; echo "E92_DONE $n" >> data/e92.log
[[ $n -ge 6 ]] || { tg "PRC E92: собрано $n из 6."; exit 1; }
o95=data/cv_preds_e95_cand; mkdir -p $o95
say "E95: фолды 1,2,3 | 4,5,6 в две полосы"
lane 1,2,3 $o95 data/e95.log $E95; lane 4,5,6 $o95 data/e95.log $E95
wait
m=$(ls $o95/fold_*.npz | wc -l | tr -d ' '); say "E95 собрано $m из 6"
[[ $m -ge 6 ]] || { tg "PRC E95: собрано $m из 6."; exit 1; }
PYTHONPATH=src .venv/bin/python - <<'PY' | tee data/e95_cv.log
from fold_metrics import compare
dec, tot, small, trim, lower, n, strict = compare("data/cv_preds_e89_cand", "data/cv_preds_e95_cand", lb_rmse=242.6556)
ok = n == 6 and -dec >= 0.3 and lower >= 4 and -trim >= 0.3
print(f"DECISION E95: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s)")
PY
tg "PRC E95 — сиды классификатора, вердикт:
$(grep 'DECISION E95' data/e95_cv.log)
Прогноз был -0.05..-0.50, центр -0.15."
say "готово"
