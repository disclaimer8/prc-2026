#!/bin/zsh
# Night 27->28.09 on the Mac (journal 122): CatBoost seeds 10-14, merge 15 seeds, blend -> v461, hetzner tail -> v467 / v468 (E145).
set -uo pipefail
cd ~/prc-2026
export CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600 PROXY_SCALE=1 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export FEATURES=data/features_e106.parquet DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
export CATBOOST_THREADS=10 OMP_NUM_THREADS=10 POLARS_MAX_THREADS=10
L=data/night_27sep_mac.log
for s in 10 11 12 13 14; do
  echo "[$(date +%H:%M)] seed $s" >> $L
  nice -n 10 .venv/bin/python src/catboost_seeds.py 5600 $s >> data/night_mac_seeds.log 2>&1 && echo "[$(date +%H:%M)] seed $s ok" >> $L || echo "[$(date +%H:%M)] seed $s FAILED" >> $L
done
.venv/bin/python - >> $L 2>&1 <<'PY'
import os, numpy as np
z = np.load("submissions/cat_reg_rank_seeds_v440.npz"); ids, P = z["mvt_id"], z["per_seed"]
extra = [np.load(f"submissions/cat_reg_rank_seed{s}.npz") for s in range(10, 15) if os.path.exists(f"submissions/cat_reg_rank_seed{s}.npz")]
assert all(np.array_equal(e["mvt_id"], ids) for e in extra)
P2 = np.vstack([P] + [e["reg"][None, :] for e in extra])
np.savez("submissions/cat_reg_rank_seeds_v460.npz", mvt_id=ids, reg=P2.mean(axis=0), per_seed=P2)
print(f"merged {P2.shape[0]} CatBoost seeds; RMS change of the average vs 10 seeds {np.sqrt(((P2.mean(0) - P.mean(0)) ** 2).mean()):.2f} s")
PY
cp -p submissions/cat_reg_rank.npz submissions/cat_reg_rank_before_night.npz
cp -f submissions/cat_reg_rank_seeds_v460.npz submissions/cat_reg_rank.npz
LGB_SEEDS=submissions/lgb_reg_seeds_v240.npz nice -n 10 .venv/bin/python src/blend.py 240 461 0.65 2>&1 | grep -E "^wrote" >> $L
cp -p submissions/cat_reg_rank_before_night.npz submissions/cat_reg_rank.npz
scp -q submissions/gentle-igloo_v461.parquet submissions/parts_v461.npz hetzner:/root/prc-2026/submissions/ && echo "[$(date +%H:%M)] v461 on hetzner" >> $L
ssh hetzner 'cd /root/prc-2026 && until grep -qE "\] done" data/night_27sep.log 2>/dev/null; do sleep 60; done; nice -n 19 taskset -c 6-11 bash src/hz_tail_on_base.sh 461 462' >> $L 2>&1
scp -q hetzner:/root/prc-2026/data/tail_v462.log data/ 2>/dev/null; cat data/tail_v462.log >> $L 2>/dev/null
scp -q hetzner:/root/prc-2026/submissions/gentle-igloo_v467.parquet hetzner:/root/prc-2026/submissions/gentle-igloo_v468.parquet submissions/ 2>/dev/null
echo "[$(date +%H:%M)] done" >> $L
