"""E211 optional stage (journal 399, DEFAULT OFF — not wired into any plan): level correction of the chain prediction by the
neighbours' label-free residual, decayed by time distance and centred within (airport, UTC day). Same construction as the CV
(scripts/e211_neighbour_level_cv.py: S1 NM-proxy residual of same-airport departures, S2 arrivals' taxi-in excess over the 2025
(ADES, runway, stand) normal, S3 = S1 within the same runway; weight exp(-|dt|/1800 s), window +-60 min, >= 2 neighbours).

    E211_BETA=data/e211_neighbour_level_20261008/beta_all_2025.json E211_SCALE=1.0 \
        .venv/bin/python src/e211_neighbour_level_apply.py <src version> <new version>

Reads submissions/gentle-igloo_v<src>.parquet, writes submissions/gentle-igloo_v<new>.parquet (same ID set and order, same dtype).
Rows touched: ADEP != LIRF-without-NM, prediction <= 7200. pred' = rint(pred + E211_SCALE * sum_k beta[airport][k] * S_k), clipped at 0.
Every other row keeps its value bit for bit. The signals use only observable quantities of the ranking file (MVT, AOBT_3, arrivals'
taxi-in) and the chain's own predictions; the arrivals' normal comes from the 2025 training files.
Evidence: CV gate passed in month pairs (-0.10 s, 5/6), but the gain was LTFM-only and reversed on every forward-in-time leg
(+0.08 … +0.16 s) — see analysis/neighbour_residual_level_20261008.md. Inclusion is the coordinator's call; the default is off.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import e211_neighbour_level_cv as E  # noqa: E402

ID, Y = E.ID, E.Y


def correction(dep: pl.DataFrame, s: np.ndarray, arr_rank: pl.DataFrame, arr_train: pl.DataFrame, beta: dict, scale: float) -> np.ndarray:
    """Signed correction per departure row of dep (row order kept); 0 where the stage does not apply."""
    arr_res = E.arrival_residuals(arr_rank, normal_from=arr_train)
    sig = E.build_signals(dep.select(ID, "ADEP_mvt", "RUNWAY_mvt", "t", "own_p"), s, arr_res)
    appl = ((sig["s"] <= 7200) & ~((sig["ADEP_mvt"] == "LIRF") & ~dep["has_nm"])).to_numpy()
    raw = E.apply_beta(sig, beta, E.SIGNALS)
    return np.where(appl, scale * raw, 0.0)


def main() -> None:
    src_v, new_v = int(sys.argv[1]), int(sys.argv[2])
    subs = os.environ.get("E211_SUBDIR", "submissions")
    beta_path = os.environ.get("E211_BETA", "data/e211_neighbour_level_20261008/beta_all_2025.json")
    scale = float(os.environ.get("E211_SCALE", "1.0"))
    ranking = os.environ.get("E211_RANKING", "data/prc-2026-datasets/ranking.parquet")
    training = sorted(glob.glob(os.environ.get("E211_TRAINING_GLOB", "data/prc-2026-datasets/training_*.parquet")))
    if not training:
        raise ValueError("no training files for the arrivals' normal")
    beta = json.loads(Path(beta_path).read_text())["beta"]
    sub = pl.read_parquet(f"{subs}/gentle-igloo_v{src_v}.parquet")
    if sub.columns != [ID, Y] or sub[Y].null_count() or sub[ID].n_unique() != sub.height:
        raise ValueError("invalid input submission")
    dep, arr_rank = E.load_raw([ranking])
    _, arr_train = E.load_raw(training)
    d = sub.with_columns(pl.col(ID).cast(pl.Int64).alias("_id")).join(dep.rename({ID: "_id"}), on="_id", how="left", maintain_order="left")
    if d["ADEP_mvt"].null_count():
        raise ValueError("submission rows missing from the ranking departures")
    s = d[Y].cast(pl.Float64).to_numpy()
    corr = correction(d, s, arr_rank, arr_train, beta, scale)
    new = np.clip(np.rint(s + corr), 0, None)
    changed = new != s
    print(f"E211: rows {sub.height:,}, corrected {int((corr != 0).sum()):,}, changed {int(changed.sum()):,}; scale {scale}; "
          f"RMS correction {np.sqrt(np.mean(corr[corr != 0] ** 2)) if (corr != 0).any() else 0:.1f} s, max |corr| {np.abs(corr).max():.1f} s; "
          f"by airport {dict(sorted(d.filter(pl.Series(corr != 0)).group_by('ADEP_mvt').len().rows()))}", flush=True)
    out = sub.with_columns(pl.Series(Y, new).cast(sub.schema[Y]))
    out.write_parquet(f"{subs}/gentle-igloo_v{new_v}.parquet")


if __name__ == "__main__":
    main()
