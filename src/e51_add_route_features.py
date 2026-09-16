"""E51: add the taxiway-graph route to the feature file (16.09).

    .venv/bin/python src/e51_add_route_features.py       data/features_e28.parquet -> data/features_e51.parquet

Joins data/taxi_route.parquet (src/taxi_route.py) on the same normalised stand key that src/geometry.py uses, adding
    route_m        distance stand -> runway start along taxiways and runways
    route_detour   that distance divided by the straight line the model already has
    route_extra_m  the difference between the two
Everything else is copied unchanged, so a comparison against data/cv_preds_nested measures exactly these three columns.
Median detour is 1.30 and its spread inside one airport is wide (LEMD 1.11 to 2.22), so this is not an airport constant.
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from geometry import norm_stand  # noqa: E402

src = os.environ.get("FEATURES", "data/features_e28.parquet")
dst = os.environ.get("FEATURES_OUT", "data/features_e51.parquet")

df = pl.read_parquet(src)
route = pl.read_parquet("data/taxi_route.parquet")
df = df.with_columns(_stand_key=pl.col("STAND_mvt").map_elements(norm_stand, return_dtype=pl.String))
df = df.join(route.rename({"stand_key": "_stand_key"}), on=["ADEP_mvt", "_stand_key", "RUNWAY_mvt"], how="left").drop("_stand_key")
df = df.with_columns(route_extra_m=pl.col("route_m") - pl.col("geo_dist_m"))
cov = df.select(pl.col("route_m").is_not_null().mean()).item()
print(f"route matched on {cov * 100:.1f}% of rows; median route {df['route_m'].median():.0f} m, "
      f"median extra over the straight line {df['route_extra_m'].median():.0f} m")
df.write_parquet(dst)
print(f"wrote {dst} rows={df.height} cols={df.width}")
