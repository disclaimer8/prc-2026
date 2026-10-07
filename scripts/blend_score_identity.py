"""Estimate a predeclared whole-file blend from two scores on the same population.

Does not optimise weights, reconstruct labels, create predictions or upload.
The identity is exact for unrounded predictions and exact input MSE scores.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

ID, Y = 'MVT_ID_mvt', 'TAXITIME_SEC_mvt'


def estimate(a, b, rmse_a, rmse_b, weight_b, score_decimals=4):
    if not 0 <= weight_b <= 1 or min(rmse_a, rmse_b) < 0:
        raise ValueError('convex fixed weight and nonnegative scores required')
    if not np.isfinite([weight_b, rmse_a, rmse_b]).all():
        raise ValueError('scores and weight must be finite')
    if a.shape != b.shape or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonempty aligned finite predictions required')
    distance = float(np.mean((a-b)**2))
    w = weight_b
    correction = w*(1-w)*distance
    mse = (1-w)*rmse_a**2 + w*rmse_b**2 - correction
    eps = 0.5*10.**(-score_decimals)
    low = (1-w)*max(0., rmse_a-eps)**2 + w*max(0., rmse_b-eps)**2 - correction
    high = (1-w)*(rmse_a+eps)**2 + w*(rmse_b+eps)**2 - correction
    if high < 0:
        raise ValueError('scores/files are inconsistent with a common target population')
    raw = (1-w)*a+w*b
    rounding_distance = float(np.sqrt(np.mean((np.rint(raw)-raw)**2)))
    return dict(rows=len(a), fixed_weight_b=w, input_scores=[rmse_a, rmse_b],
        pair_prediction_mse=distance, unrounded_blend_rmse=float(np.sqrt(max(0., mse))),
        score_precision_rmse_interval=[float(np.sqrt(max(0., low))), float(np.sqrt(high))],
        rounding_rms=rounding_distance,
        rounded_blend_rmse_bound=[max(0., float(np.sqrt(max(0., low)))-rounding_distance),
                                 float(np.sqrt(high))+rounding_distance],
        limitation='Requires both scores to cover these exact IDs with the same unweighted RMSE. Rounded-blend score is bounded, not identified. No weight optimisation.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('a', type=Path)
    p.add_argument('b', type=Path)
    p.add_argument('--rmse-a', required=True, type=float)
    p.add_argument('--rmse-b', required=True, type=float)
    p.add_argument('--weight-b', required=True, type=float)
    p.add_argument('--score-decimals', type=int, default=4, choices=range(0, 16))
    args = p.parse_args()
    frames = []
    for path in (args.a, args.b):
        frame = pl.read_parquet(path).select(ID, Y).sort(ID)
        if frame[ID].null_count() or frame[ID].n_unique() != frame.height or frame[Y].null_count():
            raise ValueError('duplicate IDs or null values: '+str(path))
        frames.append(frame)
    if not frames[0][ID].equals(frames[1][ID]):
        raise ValueError('prediction populations differ')
    result = estimate(*(f[Y].to_numpy().astype(np.float64) for f in frames),
        args.rmse_a, args.rmse_b, args.weight_b, args.score_decimals)
    result['files'] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.a,args.b)}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
