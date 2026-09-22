#!/bin/zsh
# Фолды одного плеча идут параллельно в три полосы: мак — 14 ядер, один фолд берёт ~5,5 и ~10 ГБ.
# Сначала добивается E92 (фолды 2,3,4,5,6), затем E95 (все шесть), затем судит E95. Ждать — по числу файлов.
set -uo pipefail
cd /Users/denyskolomiiets/prc-2026
say() { print -r -- "[$(date +%H:%M)] $*" }
tg() { ssh -o BatchMode=yes hetzner "set -a; . /etc/flightfinder/telegram.env; set +a; curl -s -m 20 -X POST \"https://api.telegram.org/bot\$TELEGRAM_BOT_TOKEN/sendMessage\" -d chat_id=90206238 --data-urlencode text=\"$1\" >/dev/null" 2>/dev/null; }
lane() {  # lane <folds> <out> <log> <env...>
  local folds=$1 out=$2 log=$3; shift 3
  env $@ FOLDS=$folds CV_OUT=$out OMP_NUM_THREADS=5 .venv/bin/python src/cv_fold_preds_nested.py >> $log 2>&1 &
}
E92=(NO_AOBT=1 PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 FEATURES=data/features_noaobt_e28.parquet DROP_PREFIX=day_,tq_,wxr_,month)
out=data/cv_preds_e92_noaobt_cw
say "E92: фолды 2,3 | 4,5 | 6 в три полосы"
lane 2,3 $out data/e92.log $E92; lane 4,5 $out data/e92.log $E92; lane 6 $out data/e92.log $E92
wait
n=$(ls $out/fold_*.npz | wc -l | tr -d ' '); say "E92 собрано $n из 6"; echo "E92_DONE $n" >> data/e92.log
[[ $n -ge 6 ]] || { tg "PRC E92: собрано $n из 6, партнёра нет."; exit 1; }

E95=(CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 CLF_SEEDS=5 FEATURES=data/features_e77.parquet DROP_PREFIX=day_,tq_,wxr_,month)
o95=data/cv_preds_e95_cand; mkdir -p $o95
say "E95: фолды 1,2 | 3,4 | 5,6 в три полосы"
lane 1,2 $o95 data/e95.log $E95; lane 3,4 $o95 data/e95.log $E95; lane 5,6 $o95 data/e95.log $E95
wait
m=$(ls $o95/fold_*.npz | wc -l | tr -d ' '); say "E95 собрано $m из 6"
[[ $m -ge 6 ]] || { tg "PRC E95: собрано $m из 6, вердикта нет."; exit 1; }
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
