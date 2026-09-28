#!/bin/bash
# PRC 2026 — the v447-family chain on an ARBITRARY period (draft 26.09.2026, updated 28.09 to the v447 recipe; NOT executed end to end yet).
# Audit and reasoning: analysis/final_any_period_audit_2026-09-26.md
#
# Run on hetzner:
#   RANKING=/root/new/ranking.parquet [SUBMITTING=/root/new/submitting.parquet] TAG=p1 \
#     nohup nice -n 19 taskset -c 6-11 bash /root/prc-2026/src/run_final_any_period.sh > /root/prc-2026/data/final_any.log 2>&1 &
#
# Never touches /root/prc-2026 except the markers data/final_any.{failed,done,degraded}: everything is built in WORK,
# where src/.venv/analysis and the 2025 inputs are symlinks and the new period's inputs are real files. Almost every
# script has data/prc-2026-datasets/ranking.parquet, submissions/... and data/adsblol/... baked in, so a separate
# root is what keeps Jan/Jul 2026 files (and fixed-name intermediates like cat_reg_rank_seedN.npz) out of the build.
#
# Each step is skipped when its output exists (restart after a failure resumes) and fails when its output does not
# appear. Step output goes to $WORK/logs/<step>.log; nothing is filtered through grep.
#
# Recipes (env):
#   v316-like : CAT_SEEDS=1 TABPFN_EXTRA=pfx TABPFN_SEEDS=0
#   v366-like : CAT_SEEDS=5 TABPFN_EXTRA=pfx TABPFN_SEEDS=0
#   v376-like : CAT_SEEDS=5 TABPFN_EXTRA=pfx TABPFN_SEEDS=0,1,2,3,4          (default)
#   v386-like : CAT_SEEDS=5 TABPFN_EXTRA=pfx,local_change TABPFN_SEEDS=0,1,2,3,4
#   v396-like : CAT_SEEDS=5 TABPFN_EXTRA=pfx TABPFN_SEEDS=0,1,2,3,4 E136=1
#   v447-like : CAT_SEEDS=10 TABPFN_EXTRA=pfx TABPFN_SEEDS=0 STAGE=e141 STAGE_SEEDS=42,1,2,3,4 E81=rebuild   (default since 28.09;
#               235.350 on the board). Same period as the board (Jan/Jul 2026): add TABPFN_ROWS_FROM=361:291 to keep the TabPFN
#               draw of the v366/v447 line; lirf_tabpfn.py is not deterministic run to run (journal 118).
# A second tail on the same base: rerun with the same TAG/VB and another VT (base steps are skipped).
set -Eeuo pipefail

MAIN=/root/prc-2026
: "${RANKING:?set RANKING=/path/to/the new ranking.parquet}"
SUBMITTING=${SUBMITTING:-}
TAG=${TAG:-$(date -u +%Y%m%d)}
WORK=${WORK:-/srv/prc-final-$TAG}   # /srv: 41 GB free, / only 16 GB (28.09); a WORK takes ~8 GB
VB=${VB:-900}                  # base: vVB (regimes + regressor), vVB+1 (LightGBM/CatBoost blend, parts_vVB+1)
VT=${VT:-$((VB + 10))}         # tail: vVT+2 TabPFN, +3 E14, +4 E70, +5 E64, +6 E81, +7 LOBT lift/clip = FINAL
CAT_SEEDS=${CAT_SEEDS:-10}         # any N >= 1: seed 0 = catboost_reg.py, seeds 1..N-1 = catboost_seeds.py (5 -> 10 seeds: board -0.08)
STAGE=${STAGE:-e141}               # e141 (E70 + HEAD + E141, journal 98) | e70
STAGE_SEEDS=${STAGE_SEEDS:-42,1,2,3,4}   # LightGBM seeds averaged in both ADS-B stages (E142/E143: board -0.05)
E70_DROP=${E70_DROP:-}             # E147: comma list of stage features to drop (e.g. the drifting reception counts)
E64_DROP=${E64_DROP:-}
TABPFN_ROWS_FROM=${TABPFN_ROWS_FROM:-}   # "A:B": instead of running TabPFN, copy the rows where $MAIN vA != vB (same period only)
TABPFN_EXTRA=${TABPFN_EXTRA:-pfx}
TABPFN_SEEDS=${TABPFN_SEEDS:-0}         # 1 seed matches v366 (235.524); 5 seeds = v376 (236.381, +0.857)
E136=${E136:-0}
E81=${E81:-rebuild}            # rebuild | reuse (same period: $MAIN v44/v60) | drop. Board 28.09: v314 (no E81, no lift) 237.009 vs v316 236.321 -> E81 + lift worth -0.69 s
LOBT_LIFT=${LOBT_LIFT:-1}
ADSB=${ADSB:-download}         # download | skip
ADSB_P=${ADSB_P:-3}
ADSB_ALLOW_MISSING=${ADSB_ALLOW_MISSING:-0}
WAIT_FOR=${WAIT_FOR:-}         # space-separated artifact paths to wait for before starting (never process names)
DAYS=${DAYS:-}

FAILED=$MAIN/data/final_any.failed
DONE=$MAIN/data/final_any.done
DEGRADED=$MAIN/data/final_any.degraded
PY=$MAIN/.venv/bin/python
export PY MAIN WORK
export OMP_NUM_THREADS=6 POLARS_MAX_THREADS=6 CATBOOST_THREADS=6 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage
export TABPFN_ALLOW_CPU_LARGE_DATASET=1

STEP=init
rm -f "$FAILED" "$DONE"
trap 'rc=$?; echo "[$(date -u +%FT%H:%MZ)] FAILED rc=$rc step=$STEP line=$LINENO: $BASH_COMMAND" | tee -a "$FAILED" >&2; exit $rc' ERR

log () { echo "[$(date -u +%FT%H:%MZ)] $*" | tee -a "$WORK/logs/chain.log"; }

# step NAME OUTPUT CMD... : skipped when OUTPUT exists; fails unless OUTPUT appears
step () {
  STEP=$1; local out=$2; shift 2
  if [[ -s $out ]]; then log "skip  $STEP ($out exists)"; return 0; fi
  log "start $STEP"
  "$@" > "$WORK/logs/$STEP.log" 2>&1
  [[ -s $out ]] || { echo "step $STEP ended without $out (see logs/$STEP.log)" >&2; return 1; }
  log "done  $STEP -> $out"
}
sub () { echo "submissions/gentle-igloo_v$1.parquet"; }

# ---------------------------------------------------------------- 0. wait for artifacts, then preflight
for f in $WAIT_FOR; do
  until [[ -e $f ]]; do echo "[$(date -u +%H:%MZ)] waiting for $f"; sleep 300; done
done
[[ -s $RANKING ]] || { echo "no $RANKING" >&2; false; }
[[ $E81 == drop || $E81 == rebuild || $E81 == reuse ]] || { echo "E81 must be drop|rebuild|reuse" >&2; false; }
[[ $ADSB == download || $ADSB == skip ]] || { echo "ADSB must be download|skip" >&2; false; }
[[ $ADSB == skip && $E81 == rebuild ]] && { echo "E81=rebuild needs ADS-B (E36/E42 stages of v44/v60)" >&2; false; }

mkdir -p "$WORK"/{logs,submissions,scripts,data/prc-2026-datasets,data/adsblol,data/adsblol_kt1,data/adsblol_sw_kt1,data/metar,data/metar_wind}
cd "$WORK"
STEP=recipe
# a base built with one CAT_SEEDS must not be reused silently by a run with another
if [[ -f logs/recipe_base_v$VB ]]; then
  grep -qx "CAT_SEEDS=$CAT_SEEDS" logs/recipe_base_v$VB || { echo "base v$VB was built with $(cat logs/recipe_base_v$VB); use another VB" >&2; false; }
else
  echo "CAT_SEEDS=$CAT_SEEDS" > logs/recipe_base_v$VB
fi
echo "TABPFN_EXTRA=$TABPFN_EXTRA TABPFN_SEEDS=$TABPFN_SEEDS TABPFN_ROWS_FROM=$TABPFN_ROWS_FROM E136=$E136 STAGE=$STAGE STAGE_SEEDS=$STAGE_SEEDS E70_DROP=$E70_DROP E64_DROP=$E64_DROP E81=$E81 LOBT_LIFT=$LOBT_LIFT ADSB=$ADSB" > logs/recipe_tail_v$VT
log "WORK=$WORK RANKING=$RANKING VB=$VB VT=$VT $(cat logs/recipe_base_v$VB) $(cat logs/recipe_tail_v$VT)"

STEP=preflight
for p in src .venv analysis; do [[ -e $p ]] || ln -s "$MAIN/$p" "$p"; done
[[ -e scripts/get_adsb_day.sh ]] || cp "$MAIN/scripts/get_adsb_day.sh" scripts/   # copied, not linked: it cd-s to $(dirname $0)/..
for p in ext cv_preds_nomonth; do [[ -e data/$p ]] || ln -s "$MAIN/data/$p" "data/$p"; done
for f in "$MAIN"/data/prc-2026-datasets/training_*.parquet; do ln -sf "$f" data/prc-2026-datasets/; done
cp -f "$RANKING" data/prc-2026-datasets/ranking.parquet
REF_RANKING=$MAIN/data/prc-2026-datasets/ranking.parquet REF_SUBMITTING=$MAIN/data/prc-2026-datasets/submitting.parquet \
SUBMITTING="$SUBMITTING" E81="$E81" "$PY" - > logs/preflight.log 2>&1 <<'PY'
import os, sys, glob
import polars as pl
new = pl.read_parquet("data/prc-2026-datasets/ranking.parquet")
ref = pl.read_parquet(os.environ["REF_RANKING"], n_rows=1000)
bad = [(c, str(ref.schema[c]), str(new.schema.get(c))) for c in ref.columns if new.schema.get(c) != ref.schema[c]]
extra = [c for c in new.columns if c not in ref.columns]
if bad or extra:
    sys.exit(f"schema differs from the Jan/Jul ranking: {bad}, extra {extra} (missing AOBT_3_flt needs the NO_AOBT path, REVIEW_BRIEF 21o)")
dep = new.filter(pl.col("PHASE_mvt") == "DEP")
aps = set(new["ADEP_mvt"].drop_nulls().unique()) | set(new["ADES_mvt"].drop_nulls().unique())
known = {"EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"}
dep_aps = set(dep["ADEP_mvt"].unique())
if dep_aps - known:
    sys.exit(f"departure airports outside the ten: {sorted(dep_aps - known)} (replace_strict in E36/E42/E64, ELEV_FT, AIRPORT_TZ)")
years = sorted(set(dep["MVT_TIME_UTC_mvt"].dt.year().unique().to_list()))
if years != [2026]:
    sys.exit(f"departure years {years}: stage globs are hard-coded to 2026 (e70_apply.py:77-78, e64_apply.py:64-65) and "
             f"extractors read training_* for 2025 (adsb_offblock_v2.py:128-129) — patch those first")
train_ids = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt"]) for f in glob.glob("data/prc-2026-datasets/training_*.parquet")])
ov = dep.join(train_ids, on="MVT_ID_mvt", how="inner").height
if ov:
    sys.exit(f"{ov} ranking MVT_IDs also appear in 2025")
arr = new.filter(pl.col("PHASE_mvt") == "ARR")
print(f"rows {new.height}, DEP {dep.height}, days {dep['MVT_TIME_UTC_mvt'].dt.date().n_unique()}, "
      f"{dep['MVT_TIME_UTC_mvt'].min()} .. {dep['MVT_TIME_UTC_mvt'].max()}")
print("DEP target non-null (must be 0):", int(dep["TAXITIME_SEC_mvt"].is_not_null().sum()))
print(f"ARR taxi-in null share {arr['TAXITIME_SEC_mvt'].is_null().mean():.3f} (Jan/Jul 2026: ~0); "
      f"DEP AOBT_3 null share {dep['AOBT_3_flt'].is_null().mean():.3f}; DEP no-NM share {dep['FLIGHT_ID_mvt'].is_null().mean():.3f}")
print("DEP rows by airport:", dict(sorted(dep.group_by("ADEP_mvt").len().iter_rows())))
# submission template
tpl_ref = pl.read_parquet(os.environ["REF_SUBMITTING"])
if os.environ.get("SUBMITTING"):
    tpl = pl.read_parquet(os.environ["SUBMITTING"])
    if tpl.schema != tpl_ref.schema:
        sys.exit(f"submitting schema {tpl.schema} != {tpl_ref.schema}")
else:
    if tpl_ref.columns != ["MVT_ID_mvt", "TAXITIME_SEC_mvt"]:
        sys.exit(f"cannot build a template: reference columns {tpl_ref.columns}")
    tpl = dep.select(pl.col("MVT_ID_mvt").cast(tpl_ref.schema["MVT_ID_mvt"]),
                     pl.lit(None).cast(tpl_ref.schema["TAXITIME_SEC_mvt"]).alias("TAXITIME_SEC_mvt"))
    print("template built from ranking DEP rows (no SUBMITTING given)")
if not tpl["MVT_ID_mvt"].sort().equals(dep["MVT_ID_mvt"].cast(tpl.schema["MVT_ID_mvt"]).sort()):
    sys.exit("template MVT_IDs differ from ranking DEP rows")
tpl.write_parquet("data/prc-2026-datasets/submitting.parquet")
# the 2025 stage inputs the chain reads (hetzner: offblock2 in data/adsblol; offblock3/5 there only matter for E81=rebuild)
need = {"data/cv_preds_nomonth/fold_*.npz": 6, f"{os.environ['MAIN']}/data/adsblol/offblock2_2025-*.parquet": 360,
        f"{os.environ['MAIN']}/data/adsblol/visit_2025-*.parquet": 360, f"{os.environ['MAIN']}/data/adsblol/visit_m_2025-*.parquet": 360,
        f"{os.environ['MAIN']}/data/adsblol_kt1/offblock5_2025-*.parquet": 360, f"{os.environ['MAIN']}/data/adsblol_sw_kt1/offblock5_2025-*.parquet": 360,
        "analysis/sources/samples/osm_overpass/*.json": 10, "data/ext/openflights_airlines.dat": 1}
if os.environ["E81"] == "rebuild":
    need.update({f"{os.environ['MAIN']}/data/adsblol/offblock3_2025-*.parquet": 360, f"{os.environ['MAIN']}/data/adsblol/offblock5_2025-*.parquet": 360})
short = {p: len(glob.glob(p)) for p, n in need.items() if len(glob.glob(p)) < n}
if short:
    sys.exit(f"missing inputs (found / needed): {short}")
print("preflight ok")
PY
cat logs/preflight.log

# 2025 ADS-B outputs: one symlink per file, so the stage globs see 2025 + the new days and nothing of Jan/Jul 2026
for d in adsblol adsblol_kt1 adsblol_sw_kt1; do
  for f in "$MAIN"/data/$d/offblock[235]_2025-*.parquet "$MAIN"/data/$d/visit_2025-*.parquet "$MAIN"/data/$d/visit_m_2025-*.parquet \
           "$MAIN"/data/$d/head_2025-*.parquet; do
    [[ -e $f ]] && ln -sf "$f" data/$d/
  done
done

if [[ -z $DAYS ]]; then
  DAYS=$("$PY" -c "
import polars as pl
r = pl.read_parquet('data/prc-2026-datasets/ranking.parquet', columns=['PHASE_mvt', 'MVT_TIME_UTC_mvt']).filter(pl.col('PHASE_mvt') == 'DEP')
print(' '.join(str(d) for d in sorted(r['MVT_TIME_UTC_mvt'].dt.date().unique().to_list())))")
fi
FIRST=${DAYS%% *}; LAST=${DAYS##* }
log "days: $(wc -w <<< "$DAYS") ($FIRST .. $LAST)"

# ---------------------------------------------------------------- 1. METAR for the new period
# get_external.sh has the range 2024-12-25..2026-08-02 baked in; keep the 2025 part as trained, append the period.
STEP=metar
if [[ ! -s logs/metar.ok ]]; then
  from=$(date -u -d "$FIRST -1 day" +%Y-%m-%d); to=$(date -u -d "$LAST +2 day" +%Y-%m-%d)
  rng="year1=${from:0:4}&month1=$((10#${from:5:2}))&day1=$((10#${from:8:2}))&year2=${to:0:4}&month2=$((10#${to:5:2}))&day2=$((10#${to:8:2}))&tz=Etc/UTC&format=onlycomma&latlon=no&elev=no&missing=null&trace=T&direct=no&report_type=3&report_type=4"
  for ap in EDDF EDDM EGLL EHAM LEBL LEMD LFPG LIRF LSZH LTFM; do
    for kind in metar metar_wind; do
      if [[ $kind == metar ]]; then q="data=tmpc&data=dwpc&data=sknt&data=gust&data=vsby&data=skyl1&data=wxcodes&data=metar"; else q="data=drct&data=sknt"; fi
      [[ -s $MAIN/data/$kind/$ap.csv ]] || { echo "no $MAIN/data/$kind/$ap.csv: run scripts/get_external.sh on hetzner first" >&2; false; }
      curl -sf -m 300 --retry 3 -o data/$kind/$ap.new.csv "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=$ap&$q&$rng"
      [[ $(wc -l < data/$kind/$ap.new.csv) -gt 20 ]] || { echo "METAR $kind $ap: empty answer for $from..$to" >&2; false; }
      { cat "$MAIN/data/$kind/$ap.csv"; tail -n +2 data/$kind/$ap.new.csv; } | awk '!seen[$0]++' > data/$kind/$ap.csv
      rm -f data/$kind/$ap.new.csv
    done
  done
  echo ok > logs/metar.ok; log "METAR appended $from..$to"
fi

# ---------------------------------------------------------------- 2. ADS-B for every new day
adsb_day () {  # always returns 0; a day that cannot be completed leaves logs/adsb_missing_<day>
  local d=$1 v
  cd "$WORK"
  if [[ ! -s data/adsblol/surface_$d.parquet ]]; then
    bash scripts/get_adsb_day.sh "$d" > "logs/adsb_get_$d.log" 2>&1 || { echo "download" > "logs/adsb_missing_$d"; return 0; }
  fi
  {
    for v in 2 3 5; do
      [[ -s data/adsblol/offblock${v}_$d.parquet ]] || "$PY" src/adsb_offblock_v$v.py "$d"
      [[ -s data/adsblol_kt1/offblock${v}_$d.parquet ]] || MOVING_KT=1 ADSB_OUT_DIR=data/adsblol_kt1 "$PY" src/adsb_offblock_v$v.py "$d"
      [[ -s data/adsblol_sw_kt1/offblock${v}_$d.parquet ]] || MOVING_KT=1 STAND_WALK=1 ADSB_OUT_DIR=data/adsblol_sw_kt1 "$PY" src/adsb_offblock_v$v.py "$d"
    done
    [[ -s data/adsblol/visit_$d.parquet ]] || "$PY" src/adsb_visit.py "$d"
    [[ -s data/adsblol/visit_m_$d.parquet ]] || "$PY" src/e69_visit_merge.py "$d"
    [[ -s data/adsblol/head_$d.parquet ]] || "$PY" src/e134_head_features.py "$d"   # E141 stage (HEAD of the surface track)
  } > "logs/adsb_extract_$d.log" 2>&1 || { echo "extract" > "logs/adsb_missing_$d"; return 0; }
  local f
  for f in data/adsblol/offblock{2,3,5}_$d.parquet data/adsblol_kt1/offblock{2,3,5}_$d.parquet \
           data/adsblol_sw_kt1/offblock{2,3,5}_$d.parquet data/adsblol/visit_$d.parquet data/adsblol/visit_m_$d.parquet \
           data/adsblol/head_$d.parquet; do
    [[ -s $f ]] || { echo "incomplete $f" > "logs/adsb_missing_$d"; return 0; }
  done
  rm -f "logs/adsb_missing_$d"
}
export -f adsb_day

if [[ $ADSB == download ]]; then
  STEP=adsb
  command -v curl > /dev/null || { echo "curl missing (get_adsb_day.sh streams the archive and reads the GitHub API with it)" >&2; false; }
  log "ADS-B: $(wc -w <<< "$DAYS") days, $ADSB_P in parallel"
  printf '%s\n' $DAYS | xargs -P "$ADSB_P" -I{} bash -c 'adsb_day {}'
  missing=$(ls logs/adsb_missing_* 2>/dev/null | sed 's/.*adsb_missing_//' | tr '\n' ' ' || true)
  if [[ -n $missing ]]; then
    log "ADS-B incomplete days: $missing"
    [[ $ADSB_ALLOW_MISSING == 1 ]] || { echo "ADS-B days incomplete: $missing (ADSB_ALLOW_MISSING=1 to continue degraded)" >&2; false; }
    echo "$TAG ADS-B days without stage inputs: $missing" >> "$DEGRADED"
  fi
  STEP=adsb_coverage
  "$PY" - > logs/adsb_coverage.log 2>&1 <<'PY'
import glob, polars as pl
def cov(pattern):
    return pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "coverage"]) for f in sorted(glob.glob(pattern))])
rk = pl.read_parquet("data/prc-2026-datasets/ranking.parquet", columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt"]).filter(pl.col("PHASE_mvt") == "DEP")
tr = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt"]) for f in glob.glob("data/prc-2026-datasets/training_*.parquet")]).filter(pl.col("PHASE_mvt") == "DEP")
for name, d, pat in (("2025", tr, "data/adsblol_kt1/offblock2_2025-*.parquet"), ("new", rk, "data/adsblol_kt1/offblock2_2026-*.parquet")):
    j = d.join(cov(pat), on="MVT_ID_mvt", how="left")
    print(name, dict(sorted(j.group_by("ADEP_mvt").agg((pl.col("coverage") >= 2).mean().round(3)).iter_rows())))
PY
  cat logs/adsb_coverage.log
else
  log "ADSB=skip: E70/E64 will be skipped (expected cost: the whole ADS-B stage gain)"
  echo "$TAG ADSB=skip: no ADS-B stages" >> "$DEGRADED"
fi

# ---------------------------------------------------------------- 3. features
STEP=features
step features_final data/features_final.parquet env FEATURES_OUT=data/features_final.parquet "$PY" src/features.py
step features_e28 data/features_e28.parquet env E28_IN=data/features_final.parquet E28_OUT=data/features_e28.parquet "$PY" src/e28_forward_neighbours.py build
step features_e77 data/features_e77.parquet env FEATURES=data/features_e28.parquet FEATURES_OUT=data/features_e77.parquet "$PY" src/e77_filed_twins.py
# E106 is not in the repository as a script (REVIEW_BRIEF, 23.09): airline_prefix = coalesce(NM call sign prefix,
# ^([A-Z]{3})\d of FLIGHT_mvt). Checked below against the reference features_e106 on the 2025 rows.
step features_e106 data/features_e106.parquet "$PY" -c "
import polars as pl
f = pl.read_parquet('data/features_e77.parquet')
f = f.with_columns(airline_prefix=pl.coalesce(pl.col('CALLSIGN_flt').str.extract(r'^([A-Z]{3})', 1),
                                              pl.col('FLIGHT_mvt').str.extract(r'^([A-Z]{3})\d', 1)))
f.write_parquet('data/features_e106.parquet'); print('airline_prefix null', f['airline_prefix'].null_count(), 'of', f.height)"
step features_e113 data/features_e113.parquet "$PY" src/e113_prefix_lookup.py
if [[ $E81 == rebuild ]]; then
  step features_e32 data/features_e32.parquet "$PY" src/e32_adsb_in_model.py build
fi
STEP=features_parity
[[ -s logs/features_parity.ok ]] || { REF=$MAIN/data "$PY" - > logs/features_parity.log 2>&1 <<'PY'
import os, sys, polars as pl
chk = ["MVT_ID_mvt", "airline_prefix", "d_mvt_lobt", "d_mvt_aobt", "local_hour", "nbf_ap_30_0_mean", "fw_ap_15_mean"]
for name in ("features_e106", "features_e113"):
    ref_s = pl.read_parquet_schema(f"{os.environ['REF']}/{name}.parquet")
    new_s = pl.read_parquet_schema(f"data/{name}.parquet")
    if dict(ref_s) != dict(new_s):
        sys.exit(f"{name}: schema differs from the reference: only ref {sorted(set(ref_s) - set(new_s))}, only new {sorted(set(new_s) - set(ref_s))}, "
                 f"dtype {[(c, str(ref_s[c]), str(new_s[c])) for c in ref_s if c in new_s and ref_s[c] != new_s[c]]}")
    if list(ref_s) != list(new_s):   # the model matrix follows file order, and feature_fraction samples by position
        sys.exit(f"{name}: same columns in another order; first difference at {next(i for i, (a, b) in enumerate(zip(ref_s, new_s)) if a != b)}")
    cols = [c for c in chk if c in new_s]
    ref = pl.read_parquet(f"{os.environ['REF']}/{name}.parquet", columns=cols + ["split"]).filter(pl.col("split") == "train").drop("split")
    new = pl.read_parquet(f"data/{name}.parquet", columns=cols + ["split"]).filter(pl.col("split") == "train").drop("split")
    j = ref.join(new, on="MVT_ID_mvt", how="full", suffix="_new")
    if j.height != ref.height or j.height != new.height:
        sys.exit(f"{name}: 2025 rows differ ({ref.height} ref, {new.height} new, {j.height} joined)")
    for c in cols[1:]:
        a, b = j[c], j[c + "_new"]
        if a.dtype.is_float():
            a, b = a.fill_nan(None), b.fill_nan(None)
        diff = int((a.is_null() != b.is_null()).sum() + ((a != b) & a.is_not_null() & b.is_not_null()).fill_null(False).sum())
        print(f"{name}.{c}: 2025 rows differing from the reference {diff}")
        if diff and c in ("airline_prefix", "d_mvt_lobt", "d_mvt_aobt", "local_hour"):
            sys.exit(f"{name}.{c} does not reproduce on 2025 rows")
print("parity ok")
PY
  echo ok > logs/features_parity.ok; }
cat logs/features_parity.log

# ---------------------------------------------------------------- 4. base model (E116 configuration)
export FEATURES=data/features_e106.parquet DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config
export PROXY_SCALE=1 CLF_COST_WEIGHT=1 CLF_COST_MIN_SPREAD=3600
step validate data/validation_v$VB.json bash -c "'$PY' src/train.py validate && cp data/last_validation.json data/validation_v$VB.json"
cp data/validation_v$VB.json data/last_validation.json   # train.py submit reads this fixed path
step submit "$(sub $VB)" "$PY" src/train.py submit $VB
REG=$("$PY" -c "import json;print(int(json.load(open('data/validation_v$VB.json'))['best']['reg']*1.1))")
step lgb_seeds submissions/lgb_reg_seeds_v$VB.npz bash -c "'$PY' src/seed_avg.py rank 5 $REG && cp submissions/lgb_reg_seeds.npz submissions/lgb_reg_seeds_v$VB.npz"
step catboost0 submissions/cat_reg_rank_v$VB.npz bash -c "'$PY' src/catboost_reg.py 5600 && cp submissions/cat_reg_rank.npz submissions/cat_reg_rank_v$VB.npz"
if [[ $CAT_SEEDS -gt 1 ]]; then
  [[ -e submissions/cat_reg_rank_seed0.npz ]] || cp submissions/cat_reg_rank_v$VB.npz submissions/cat_reg_rank_seed0.npz
  EXTRA_SEEDS=$(seq -s ' ' 1 $((CAT_SEEDS - 1))); ALL_SEEDS=$(seq -s ' ' 0 $((CAT_SEEDS - 1)))
  step catboost_seeds submissions/cat_reg_rank_seeds_v$VB.npz bash -c \
    "'$PY' src/catboost_seeds.py 5600 $EXTRA_SEEDS && '$PY' src/catboost_seeds.py merge $ALL_SEEDS && cp submissions/cat_reg_rank_seeds.npz submissions/cat_reg_rank_seeds_v$VB.npz"
  CAT=submissions/cat_reg_rank_seeds_v$VB.npz
elif [[ $CAT_SEEDS == 1 ]]; then
  CAT=submissions/cat_reg_rank_v$VB.npz
else
  STEP=catboost_seeds; echo "CAT_SEEDS must be >= 1" >&2; false
fi
cp -f "$CAT" submissions/cat_reg_rank.npz   # blend.py reads this fixed path
step blend "$(sub $((VB + 1)))" env LGB_SEEDS=submissions/lgb_reg_seeds_v$VB.npz "$PY" src/blend.py $VB $((VB + 1)) 0.65
B=$((VB + 1)); PARTS=submissions/parts_v$B.npz
[[ -s $PARTS ]] || { STEP=blend; echo "no $PARTS" >&2; false; }

# ---------------------------------------------------------------- 5. tail: TabPFN -> E14 -> E70 -> E64
TAB_ENV=(TABPFN_EXTRA="$TABPFN_EXTRA" TABPFN_SEEDS="$TABPFN_SEEDS" PFX_FEATURES=data/features_e113.parquet)
[[ $E136 == 1 ]] && TAB_ENV+=(TABPFN_BINS="<2h,2-4h,4-8h,8-12h,12-30h" P_DAY_MIN_D=43200)
if [[ -n $TABPFN_ROWS_FROM ]]; then   # keep the TabPFN draw of an earlier file on the same period (journal 118)
  step tabpfn "$(sub $((VT + 2)))" env A=${TABPFN_ROWS_FROM%%:*} Z=${TABPFN_ROWS_FROM##*:} B=$B OUT=$((VT + 2)) "$PY" -c "
import os, polars as pl
Y = 'TAXITIME_SEC_mvt'; M = os.environ['MAIN']
r = lambda p: pl.read_parquet(p)
a, z, s = r(f\"{M}/submissions/gentle-igloo_v{os.environ['A']}.parquet\"), r(f\"{M}/submissions/gentle-igloo_v{os.environ['Z']}.parquet\"), r(f\"submissions/gentle-igloo_v{os.environ['B']}.parquet\")
assert set(a['MVT_ID_mvt'].to_list()) == set(s['MVT_ID_mvt'].to_list()), 'TABPFN_ROWS_FROM needs the same ranking rows'
chg = a.join(z, on='MVT_ID_mvt', suffix='_z').filter(pl.col(Y) != pl.col(f'{Y}_z')).select('MVT_ID_mvt', pl.col(Y).alias('tab'))
s.join(chg, on='MVT_ID_mvt', how='left').with_columns(pl.coalesce('tab', Y).alias(Y)).select(s.columns).write_parquet(f\"submissions/gentle-igloo_v{os.environ['OUT']}.parquet\")
print(f'wrote v{os.environ[\"OUT\"]}: {chg.height} TabPFN rows copied from v{os.environ[\"A\"]}')
"
else
  step tabpfn "$(sub $((VT + 2)))" env "${TAB_ENV[@]}" "$PY" src/lirf_tabpfn.py $B $((VT + 2))
fi
grep -E "prefix codes|wrote" logs/tabpfn.log | tee -a logs/chain.log || true
step e14 "$(sub $((VT + 3)))" "$PY" src/e14_apply.py $((VT + 2)) $B $((VT + 3))
if [[ $ADSB == download ]]; then
  STAGE_PY=src/e141_apply.py; [[ $STAGE == e70 ]] && STAGE_PY=src/e70_apply.py
  step e70 "$(sub $((VT + 4)))" env STAGE_SEEDS=$STAGE_SEEDS DROP_FEATS=$E70_DROP ADSB_DIR=data/adsblol_kt1 WALK_DIR=data/adsblol_sw_kt1 PARTS=$PARTS "$PY" $STAGE_PY $((VT + 3)) $((VT + 4))
  step e64 "$(sub $((VT + 5)))" env STAGE_SEEDS=$STAGE_SEEDS DROP_FEATS=$E64_DROP ADSB_DIR=data/adsblol_kt1 PARTS=$PARTS "$PY" src/e64_apply.py $((VT + 4)) $((VT + 5))
else
  step e70 "$(sub $((VT + 4)))" cp "$(sub $((VT + 3)))" "$(sub $((VT + 4)))"
  step e64 "$(sub $((VT + 5)))" cp "$(sub $((VT + 4)))" "$(sub $((VT + 5)))"
fi

# ---------------------------------------------------------------- 6. E81 submission blend
if [[ $E81 == rebuild ]]; then
  # partners in their own root: they rewrite cat_reg_rank.npz, lgb_reg_seeds.npz and data/last_validation.json
  W2=$WORK/e81
  mkdir -p "$W2"/{logs,submissions,data}
  for p in src .venv analysis; do [[ -e $W2/$p ]] || ln -s "$MAIN/$p" "$W2/$p"; done
  for p in prc-2026-datasets ext cv_preds_nomonth adsblol adsblol_kt1 adsblol_sw_kt1 features_e28.parquet features_e32.parquet; do
    [[ -e $W2/data/$p ]] || ln -s "$WORK/data/$p" "$W2/data/$p"
  done
  P44=$((VB + 40)); P60=$((VB + 56))
  (
    set -Eeuo pipefail
    cd "$W2"
    unset PROXY_SCALE CLF_COST_WEIGHT CLF_COST_MIN_SPREAD TABPFN_EXTRA TABPFN_SEEDS TABPFN_BINS P_DAY_MIN_D ADSB_DIR WALK_DIR PFX_FEATURES
    wstep () { STEP=$1; local out=$2; shift 2; [[ -s $out ]] && return 0; "$@" > "logs/$STEP.log" 2>&1; [[ -s $out ]]; }
    # v44 = per-airport models + TabPFN + E14 + E42 stage (src/run_final_e38.sh, src/after_v42_stage.sh), 3 kt ADS-B
    export FEATURES=data/features_e28.parquet DROP_PREFIX=day_,tq_,wxr_,month
    wstep p44_models "$(sub $P44)" "$PY" src/submit_per_airport.py $P44
    wstep p44_tabpfn "$(sub $((P44 + 1)))" "$PY" src/lirf_tabpfn.py $P44 $((P44 + 1))
    wstep p44_e14 "$(sub $((P44 + 2)))" "$PY" src/e14_apply.py $((P44 + 1)) $P44 $((P44 + 2))
    wstep p44_e42 "$(sub $((P44 + 4)))" env PARTS=submissions/parts_v$P44.npz "$PY" src/e42_stand_geometry_per_airport.py apply $((P44 + 2)) $((P44 + 4))
    # v60 = E32 features + E46 proxy scale, full chain + E36 stage (src/run_final_e52.sh), 3 kt ADS-B
    export FEATURES=data/features_e32.parquet DROP_PREFIX=day_,tq_,wxr_,month PROXY_SCALE=1
    wstep p60_validate data/validation_v$P60.json bash -c "'$PY' src/train.py validate && cp data/last_validation.json data/validation_v$P60.json"
    cp data/validation_v$P60.json data/last_validation.json
    wstep p60_submit "$(sub $P60)" "$PY" src/train.py submit $P60
    R60=$("$PY" -c "import json;print(int(json.load(open('data/validation_v$P60.json'))['best']['reg']*1.1))")
    wstep p60_seeds submissions/lgb_reg_seeds_v$P60.npz bash -c "'$PY' src/seed_avg.py rank 5 $R60 && cp submissions/lgb_reg_seeds.npz submissions/lgb_reg_seeds_v$P60.npz"
    wstep p60_cat submissions/cat_reg_rank_v$P60.npz bash -c "'$PY' src/catboost_reg.py 5600 && cp submissions/cat_reg_rank.npz submissions/cat_reg_rank_v$P60.npz"
    cp -f submissions/cat_reg_rank_v$P60.npz submissions/cat_reg_rank.npz
    wstep p60_blend "$(sub $((P60 + 1)))" env LGB_SEEDS=submissions/lgb_reg_seeds_v$P60.npz "$PY" src/blend.py $P60 $((P60 + 1)) 0.65
    wstep p60_tabpfn "$(sub $((P60 + 2)))" "$PY" src/lirf_tabpfn.py $((P60 + 1)) $((P60 + 2))
    wstep p60_e14 "$(sub $((P60 + 3)))" "$PY" src/e14_apply.py $((P60 + 2)) $((P60 + 1)) $((P60 + 3))
    wstep p60_e36 "$(sub $((P60 + 4)))" env PARTS=submissions/parts_v$((P60 + 1)).npz "$PY" src/e36_adsb_stand_geometry.py apply $((P60 + 3)) $((P60 + 4))
  )   # a failure inside stops the run through the inherited ERR trap (no "||" here: it would switch set -e off inside)
  ln -sf "$W2/$(sub $((P44 + 4)))" "$(sub $((P44 + 4)))"
  ln -sf "$W2/$(sub $((P60 + 4)))" "$(sub $((P60 + 4)))"
  step e81 "$(sub $((VT + 6)))" env BLEND_BASE=$((VT + 5)) BLEND_PARTS="$((P44 + 4)):0.1214,$((P60 + 4)):0.0270" \
    "$PY" src/blend_submissions.py $((VT + 6))
elif [[ $E81 == reuse ]]; then
  # same period only: the E81 partners v44 / v60 of the board line, checked against this ranking's rows
  P44=$((VB + 40)); P60=$((VB + 56))
  "$PY" -c "
import polars as pl, sys
ids = set(pl.read_parquet('$(sub $((VT + 5)))')['MVT_ID_mvt'].to_list())
for v in (44, 60):
    assert set(pl.read_parquet('$MAIN/submissions/gentle-igloo_v%d.parquet' % v)['MVT_ID_mvt'].to_list()) == ids, 'E81=reuse needs the same ranking rows'
"
  ln -sf "$MAIN/submissions/gentle-igloo_v44.parquet" "$(sub $((P44 + 4)))"
  ln -sf "$MAIN/submissions/gentle-igloo_v60.parquet" "$(sub $((P60 + 4)))"
  step e81 "$(sub $((VT + 6)))" env BLEND_BASE=$((VT + 5)) BLEND_PARTS="$((P44 + 4)):0.1214,$((P60 + 4)):0.0270" \
    "$PY" src/blend_submissions.py $((VT + 6))
else
  log "E81=drop: no submission blend (expected cost ~ +0.69 s: board 28.09, v314 vs v316)"
  step e81 "$(sub $((VT + 6)))" cp "$(sub $((VT + 5)))" "$(sub $((VT + 6)))"
fi

# ---------------------------------------------------------------- 7. LOBT edge lift + window clip
if [[ $LOBT_LIFT == 1 ]]; then
  step lobt "$(sub $((VT + 7)))" env LOBT_CLIP=1 "$PY" src/lobt_edge_lift.py $((VT + 6)) $((VT + 7))
else
  step lobt "$(sub $((VT + 7)))" "$PY" src/lobt_window_clip.py $((VT + 6)) $((VT + 7))
fi
FINAL=$WORK/$(sub $((VT + 7)))

# ---------------------------------------------------------------- 8. final checks and size of every stage
STEP=final_check
VB=$VB VT=$VT "$PY" - > logs/final_check.log 2>&1 <<'PY'
import os, sys
import numpy as np, polars as pl
VB, VT = int(os.environ["VB"]), int(os.environ["VT"])
T = "TAXITIME_SEC_mvt"
tpl = pl.read_parquet("data/prc-2026-datasets/submitting.parquet")
rd = lambda v: pl.read_parquet(f"submissions/gentle-igloo_v{v}.parquet").sort("MVT_ID_mvt")
fin = rd(VT + 7)
assert fin.schema == tpl.schema, (fin.schema, tpl.schema)
assert fin["MVT_ID_mvt"].equals(tpl["MVT_ID_mvt"].sort()), "MVT_ID set differs from the template"
assert fin[T].null_count() == 0 and int(fin[T].min()) >= 0, "null or negative predictions"
f = pl.read_parquet("data/features_e106.parquet", columns=["MVT_ID_mvt", "split", "d_mvt_lobt"]).filter(pl.col("split") == "rank")
j = fin.join(f, on="MVT_ID_mvt", how="left")
out = ((j[T] < j["d_mvt_lobt"] - 3606) | (j[T] > j["d_mvt_lobt"] + 3606)).fill_null(False).sum()
print(f"final v{VT + 7}: rows {fin.height}, mean {fin[T].mean():.1f}, max {fin[T].max()}, outside the LOBT window {out}")
prev = rd(VB + 1)
for name, v in (("TabPFN", VT + 2), ("E14", VT + 3), ("E70", VT + 4), ("E64", VT + 5), ("E81", VT + 6), ("LOBT", VT + 7)):
    cur = rd(v)
    d = cur[T].cast(float).to_numpy() - prev[T].cast(float).to_numpy()
    print(f"{name:6s} v{v}: changed {int((d != 0).sum())} rows, RMS change {np.sqrt((d ** 2).mean()):.3f} s, max |d| {np.abs(d).max():.0f} s")
    prev = cur
PY
cat logs/final_check.log
echo "$FINAL" > "$DONE"
log "FINAL $FINAL (upload from the mac with UPLOAD_REF pointing at a file of the same period, e.g. v$((VB + 1)))"
