import polars as pl, glob
files = sorted(glob.glob("data/prc-2026-datasets/training_2025-*.parquet"))
df = pl.read_parquet(files[6])
print("training file", files[6], df.shape)
for c in df.columns:
    s = df[c]
    nn = s.null_count()
    ex = s.drop_nulls().head(3).to_list()
    nu = s.n_unique()
    print(f"{c:24s} {str(s.dtype):28s} nulls={nn:7d} ({nn/len(s):.1%}) nuniq={nu:7d} ex={ex}")
for c in ["PHASE_mvt","FLIGHT_RULE_mvt","MARKET_SEGMENT_flt","FLIGHT_TYPE_flt","WK_TBL_CAT_flt","FLIGHT_RULE_flt"]:
    print(df[c].value_counts().sort("count",descending=True))
r = pl.read_parquet("data/prc-2026-datasets/ranking.parquet")
print("ranking", r.shape, r.columns == df.columns)
for c in r.columns:
    s = r[c]; print(f"{c:24s} nulls={s.null_count()/len(s):.1%}")
s = pl.read_parquet("data/prc-2026-datasets/submitting.parquet")
print("submitting", s.shape, s.columns)
