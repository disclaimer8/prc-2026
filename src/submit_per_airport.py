"""E38 accepted (15.09, -10.98 s on 6/6 folds): build a submission with a separate model per airport.

    FEATURES=data/features_e28.parquet DROP_PREFIX=day_,tq_,wxr_,month .venv/bin/python src/submit_per_airport.py <version>
For every airport: rounds are chosen on January+July 2025 of that airport (as train.py validate does globally), then the
regime classifier and the residual regressor are refit on all 2025 rows of that airport with rounds x 1.1 and applied to its
ranking rows. Mirrors train.py submit otherwise: same mixture, LOBT mask and clip; writes submissions/parts_v<version>.npz
(mvt_id, proba, values, reg) and submissions/gentle-igloo_v<version>.parquet.
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import polars as pl

import train as T

version = int(sys.argv[1])
df = pl.read_parquet(T.FEATURES_PATH)
train_all, rank_all = df.filter(pl.col("split") == "train"), df.filter(pl.col("split") == "rank")
del df
ids, preds, probas, values, regs = [], [], [], [], []
for ap in sorted(rank_all["ADEP_mvt"].unique().to_list()):
    tr, rk = train_all.filter(pl.col("ADEP_mvt") == ap), rank_all.filter(pl.col("ADEP_mvt") == ap)
    is_val = pl.col("month").is_in([1, 7])
    fit, val = tr.filter(~is_val), tr.filter(is_val)
    _, _, best, _ = T.fit_predict(fit, val, T.feature_columns(fit), None, val_y=val[T.TARGET].to_numpy())
    rounds = {k: int(v * 1.1) for k, v in best.items()}
    pred, parts, _, _ = T.fit_predict(tr, rk, T.feature_columns(tr), rounds)
    ids.append(rk["MVT_ID_mvt"].to_numpy())
    preds.append(pred)
    probas.append(parts["proba"])
    values.append(parts["values"])
    regs.append(parts["reg"])
    print(f"{ap}: rows {rk.height}, rounds {rounds}, mean pred {pred.mean():.0f}", flush=True)

mvt_id = np.concatenate(ids)
np.savez(f"submissions/parts_v{version}.npz", mvt_id=mvt_id, proba=np.concatenate(probas), values=np.concatenate(values), reg=np.concatenate(regs))
template = pl.read_parquet("data/prc-2026-datasets/submitting.parquet")
sub = template.select("MVT_ID_mvt").join(pl.DataFrame({"MVT_ID_mvt": mvt_id, T.TARGET: np.concatenate(preds)}), on="MVT_ID_mvt", how="left")
assert sub.height == template.height and sub[T.TARGET].null_count() == 0
sub = sub.with_columns(pl.col(T.TARGET).round(0).cast(template.schema[T.TARGET]))
sub.write_parquet(f"submissions/gentle-igloo_v{version}.parquet")
print(f"wrote submissions/gentle-igloo_v{version}.parquet rows={sub.height} mean={sub[T.TARGET].mean():.0f} max={sub[T.TARGET].max()}")
