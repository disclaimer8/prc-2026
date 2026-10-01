"""E79: seed-average the CatBoost regressor, the way the LightGBM one already is (20.09).

    FEATURES=data/features_e28.parquet .venv/bin/python src/catboost_seeds.py 5600 1 2 3 4
    -> submissions/cat_reg_rank_seed<N>.npz per seed, then src/catboost_seeds.py merge

`src/catboost_reg.py` fits one model at `random_seed=0` and `src/blend.py` gives it weight 0.65, while the
LightGBM half of the same blend is already averaged over five seeds (`src/seed_avg.py`). That asymmetry is
free to close: seed averaging adds no capacity, it only removes the fitting noise of a single draw, and
capacity-free changes are the ones that have transferred to the board every time (REVIEW_BRIEF: E71, E64
population growth; against E32/E38/E50, which learned and did not transfer).

The gain is arithmetic once the spread is known. Averaging k seeds removes (1 - 1/k) of the seed variance;
the LightGBM measurement of the same quantity was 521 s^2 per row (sd 22.8 s), which is where the recorded
227.65 -> 226.76 came from. This run measures the CatBoost equivalent rather than assuming it, and writes
one file per seed so that an interrupted run is still usable.

Each seed is an independent fit of about an hour, so this is a job for an idle machine, not for the one
holding the experiment queue.
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")

import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))

OUT = os.environ.get("CAT_SEED_OUT", "submissions/cat_reg_rank_seed{}.npz")   # E175: another table must not overwrite the E106 seeds
MERGED = os.environ.get("CAT_MERGE_OUT", "submissions/cat_reg_rank_seeds.npz")


def merge(seeds: list[int]) -> None:
    files = [OUT.format(s) for s in seeds]
    have = [(s, f) for s, f in zip(seeds, files) if os.path.exists(f)]
    if not have:
        sys.exit("no per-seed files to merge")
    ref = np.load(have[0][1])
    ids = ref["mvt_id"]
    preds = []
    for s, f in have:
        d = np.load(f)
        if not np.array_equal(d["mvt_id"], ids):
            sys.exit(f"seed {s}: mvt_id order differs from seed {have[0][0]}")
        preds.append(d["reg"])
    P = np.vstack(preds)
    spread = float(np.mean(np.var(P, axis=0, ddof=1))) if len(P) > 1 else float("nan")
    print(f"seeds merged: {[s for s, _ in have]}")
    print(f"per-row seed variance {spread:.1f} s^2 (sd {np.sqrt(spread):.1f} s); "
          f"averaging {len(P)} removes {1 - 1 / len(P):.2f} of it")
    np.savez(MERGED, mvt_id=ids, reg=P.mean(axis=0), per_seed=P)
    print(f"wrote {MERGED} over {len(P)} seeds, mean {float(P.mean()):.1f}")


def fit_one(iterations: int, seed: int) -> None:
    from catboost import CatBoostRegressor, Pool

    import train as T
    from features import CATEGORICAL, TARGET
    from target_encoding import add_encodings

    df = pl.read_parquet(T.FEATURES_PATH)
    train, rank = df.filter(pl.col("split") == "train"), df.filter(pl.col("split") == "rank")
    lab = T.regime_label(train)
    train, rank = add_encodings(train, rank, lab == 0, fit_labels=lab)
    cols = T.feature_columns(train)
    cat_cols = [c for c in CATEGORICAL if c in cols]

    def pool(d, y=None):
        pdf = d.select(cols).to_pandas()
        for c in cat_cols:
            pdf[c] = pdf[c].fillna("NA").astype(str)
        return Pool(pdf, y, cat_features=cat_cols)

    fit = train.filter(pl.Series(lab == 0))
    model = CatBoostRegressor(iterations=iterations, learning_rate=0.08, depth=8, loss_function="RMSE",
                              thread_count=int(os.environ.get("CATBOOST_THREADS", "8")),
                              verbose=500, random_seed=seed)
    base_f, base_p = T.proxy_base(fit, rank)
    model.fit(pool(fit, np.clip(np.clip(fit[TARGET].to_numpy(), *T.REG_CLIP) - base_f, -7200, 7200)))
    reg = model.predict(pool(rank)) + base_p
    np.savez(OUT.format(seed), mvt_id=rank["MVT_ID_mvt"].to_numpy(), reg=reg)
    print(f"seed {seed}: wrote {OUT.format(seed)}, {len(reg)} rows, mean {float(np.mean(reg)):.1f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge([int(x) for x in sys.argv[2:]])
    else:
        it = int(sys.argv[1])
        for s in (int(x) for x in sys.argv[2:]):
            if os.path.exists(OUT.format(s)):
                print(f"seed {s}: already present, skipping", flush=True)
                continue
            fit_one(it, s)
