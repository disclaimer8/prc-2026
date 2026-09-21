#!/bin/zsh
# E82: wait for all three machines, collect the folds, judge by the registered gate.
#   мак фолды 1,2,3 · hetzner фолд 4 · home-pc фолды 5,6 — оба плеча на своей машине
set -uo pipefail
cd "${0:a:h}/.."
say() { print -r -- "[$(date +%H:%M)] $*" }

for i in $(seq 1 200); do
  scp -q hetzner:/root/prc-2026/data/cv_preds_e84_ctrl/fold_4_10.npz data/cv_preds_e84_ctrl/ 2>/dev/null
  scp -q hetzner:/root/prc-2026/data/cv_preds_e84_cand/fold_4_10.npz data/cv_preds_e84_cand/ 2>/dev/null
  for arm in ctrl cand; do
    # scp не раскрывает шаблоны в пути Windows — тянем каталог целиком
    rm -rf /tmp/e84hp_$arm && scp -qr "home-pc-lan:prc-2026/data/cv_preds_e84_${arm}_hp" /tmp/e84hp_$arm 2>/dev/null
    [[ -d /tmp/e84hp_$arm ]] && cp -n /tmp/e84hp_$arm/*.npz data/cv_preds_e84_$arm/ 2>/dev/null
  done
  c=$(ls data/cv_preds_e84_ctrl/fold_*.npz 2>/dev/null | wc -l | tr -d ' ')
  d=$(ls data/cv_preds_e84_cand/fold_*.npz 2>/dev/null | wc -l | tr -d ' ')
  say "собрано ctrl=$c cand=$d из 6"
  [[ "$c" -ge 6 && "$d" -ge 6 ]] && break
  sleep 120
done

c=$(ls data/cv_preds_e84_ctrl/fold_*.npz 2>/dev/null | wc -l | tr -d ' ')
d=$(ls data/cv_preds_e84_cand/fold_*.npz 2>/dev/null | wc -l | tr -d ' ')
[[ "$c" -ge 6 && "$d" -ge 6 ]] || { say "НЕ СУЖУ: собрано только ctrl=$c cand=$d"; exit 1; }

PYTHONPATH=src .venv/bin/python - <<'PY' | tee data/e84_cv.log
from fold_metrics import compare
dec, tot, small, trim, lower, n, strict = compare("data/cv_preds_e84_ctrl", "data/cv_preds_e84_cand", lb_rmse=243.8347)
ok = n == 6 and -dec >= 0.3 and lower >= 4 and -small >= 0.3 and -trim >= 0.3
print(f"DECISION E82: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s)")
PY
say "готово"
