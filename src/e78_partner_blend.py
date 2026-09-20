"""E78: blend a second full model, trained without the AOBT_3 link, into the accepted chain (20.09).

    NO_AOBT=1 FEATURES=data/features_noaobt_e28.parquet DROP_PREFIX=day_,tq_,wxr_,month PROXY_SCALE=1 \\
        .venv/bin/python src/train.py submit 90          # the partner's ranking predictions
    .venv/bin/python src/e78_partner_blend.py 91         # v82 + partner + v44 -> v91

E68 priced ABSTAINING from AOBT_3 as a replacement: +5.99 s, and the line was closed. Read the other way
round, the same arm is a partner rather than a replacement. Two full models with different proxy chains make
different errors (OOF residual correlation 0.83-0.89), and a mixture of them is variance reduction, which no
regressor can perform on itself -- so the absorption rule does not apply here the way it does to a feature.

Measured on the saved folds (LOFO-NNLS, weights fitted on five folds and applied to the sixth, decided rows):

    partner alone          decided -0.759 (6/6) | strict -0.742 | y<=7200 -0.752 | trimmed -0.745
    v44 + v60 (in v89)     decided -0.565 (6/6) | strict -0.286 | y<=7200 -0.281 | trimmed -0.514
    partner + v44 + v60    decided -0.800 (6/6) | strict -0.663 | y<=7200 -0.669 | trimmed -0.776
    partner + v44          decided -0.804 (6/6) | strict -0.659 | y<=7200 -0.665 | trimmed -0.781

The gains overlap almost entirely -- separately -0.759 and -0.565, together -0.800, not -1.32 -- so the
partner subsumes most of what v44 and v60 carried, and v60's weight collapses from 0.053 to 0.025 in its
presence. This composition passes every leg of the standing gate, which the v89 pair did not (y<=7200 was
-0.281 there against a 0.3 bar).

The blend is applied on the strict-decided rows only and v82 is kept elsewhere: the LIRF tail table and the
E14 population are written by later stages whose dilution nothing has measured. E70 and E64 rows ARE blended,
as they were in the CV measurement, where the control (data/cv_preds_e74) is itself stage-corrected output.

REGISTERED PREDICTION, WRITTEN BEFORE THE UPLOAD: 246.70, band 246.50 .. 246.95 against v89 = 246.9205.
The centre is not the CV number: CV improves by -0.24 against v89's composition, and the single board/CV
ratio we hold for a blend (v89: -0.83 on the board for -0.565 in CV, 1.47x) would put this at 246.57, but one
ratio is not a law and restricting the blend to strict-decided rows can only shrink the effect. Unlike E76
there is no identity to lean on -- the partner file has no board score, and buying one would cost a slot and
be exactly the probing we decline.
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl
from scipy.optimize import nnls

sys.path.insert(0, os.path.dirname(__file__))
import fold_metrics as FM  # noqa: E402

T = "TAXITIME_SEC_mvt"
BASE = 82
PARTNERS = {"noaobt": 90, "e38": 44}          # OOF arm -> the submission file that realises it
FOLDS = ["fold_1_7", "fold_2_8", "fold_3_9", "fold_4_10", "fold_5_11", "fold_6_12"]
REF = "data/cv_preds_e74"


def weights() -> dict:
    """NNLS on the decided rows of all six folds, the production counterpart of the LOFO check."""
    dec = FM.decided_map()
    R = {f: np.load(f"{REF}/{f}.npz") for f in FOLDS}

    def arm(d, f):
        e = np.load(f"data/cv_preds_{d}/{f}.npz")
        o = {m: i for i, m in enumerate(e["mvt_id"])}
        return e["pred"][np.array([o[m] for m in R[f]["mvt_id"]])]

    A, b = [], []
    for f in FOLDS:
        m = np.array([dec[k] for k in R[f]["mvt_id"]])
        A.append(np.column_stack([arm(d, f)[m] - R[f]["pred"][m] for d in PARTNERS]))
        b.append(-(R[f]["pred"][m] - R[f]["y"][m]))
    w, _ = nnls(np.vstack(A), np.concatenate(b))
    return dict(zip(PARTNERS, (float(x) for x in w)))


def main() -> None:
    out = int(sys.argv[1])
    w = weights()
    print("weights: " + ", ".join(f"{d}->v{PARTNERS[d]} {w[d]:.4f}" for d in PARTNERS))

    base = pl.read_parquet(f"submissions/gentle-igloo_v{BASE}.parquet")
    p = base[T].to_numpy().astype(float)
    blend = p.copy()
    for d, v in PARTNERS.items():
        q = base.select("MVT_ID_mvt").join(pl.read_parquet(f"submissions/gentle-igloo_v{v}.parquet"), on="MVT_ID_mvt", how="left")
        if q[T].null_count():
            sys.exit(f"v{v}: {q[T].null_count()} rows missing against v{BASE}")
        blend += w[d] * (q[T].to_numpy().astype(float) - p)

    strict = FM.decided_map(strict=True)
    keep = np.array([not strict.get(m, True) for m in base["MVT_ID_mvt"].to_numpy()])
    blend[keep] = p[keep]
    print(f"rows kept at v{BASE} (tail table and E14): {int(keep.sum())} of {len(p)}")

    res = base.with_columns(pl.Series(T, np.rint(blend)).cast(pl.Int32))
    if res.height != base.height or res[T].null_count():
        sys.exit("row count or null check failed")
    print(f"rows {res.height}, mean {res[T].mean():.1f}, max {res[T].max()}, "
          f"changed against v{BASE} {int((res[T].to_numpy() != base[T].to_numpy()).sum())}, "
          f"RMS change {float(np.sqrt(np.mean((res[T].to_numpy() - p) ** 2))):.2f} s")
    res.write_parquet(f"submissions/gentle-igloo_v{out}.parquet")
    print(f"FINAL submissions/gentle-igloo_v{out}.parquet")


if __name__ == "__main__":
    main()
