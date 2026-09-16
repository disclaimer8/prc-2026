"""E44 re-extraction driver (cross-platform): days of the given months whose surface file has no staleness flag yet.

    python src/reextract_months.py 2025-07 2025-08 ...
Replaces a day only after the new extraction succeeds; offblock2/3/5 of that day are rebuilt from the new points.
"""
import glob
import os
import subprocess
import sys

import polars as pl

for ym in sys.argv[1:]:
    for f in sorted(glob.glob(f"data/adsblol/surface_{ym}-*.parquet")):
        day = os.path.basename(f)[len("surface_"):-len(".parquet")]
        if "stale" in pl.read_parquet_schema(f):
            continue
        backup = f"data/adsblol/.old_surface_{day}.parquet"
        os.replace(f, backup)
        try:
            subprocess.run([sys.executable, "src/adsb_fetch_day.py", day], check=True)
            os.remove(backup)
            print(f"{day} re-extracted", flush=True)
        except Exception as e:  # noqa: BLE001 — keep the old file whatever went wrong
            if os.path.exists(backup):
                os.replace(backup, f)
            print(f"{day} FAILED ({e}), old file kept", flush=True)
print("reextract done", flush=True)
