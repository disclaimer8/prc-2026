"""Is every day the ADS-B stage will read covered by staleness features? (17.09)

    .venv/bin/python src/check_stale_ready.py

Counts against the canonical day list — the offblock2 files the stage actually globs — not against whatever
surface files happen to exist. A day being re-extracted has its surface file moved aside for a few minutes, and
a glob over surface_* then reports "nothing missing" because the missing day simply drops out of the list; that
false all-clear cost a wasted E44 launch on 17.09.
"""
import glob
import os
import sys

days = sorted(os.path.basename(f)[10:20] for f in glob.glob("data/adsblol/offblock2_*.parquet"))
missing = [d for d in days if not os.path.exists(f"data/adsblol/stale_{d}.parquet")]
rank = [d for d in missing if d.startswith("2026")]
print(f"days the stage reads: {len(days)}; with staleness features: {len(days) - len(missing)}; missing: {len(missing)}")
if missing:
    print("  missing:", ", ".join(missing[:12]) + (" ..." if len(missing) > 12 else ""))
    if rank:
        print(f"  of them in the ranking months: {', '.join(rank)} — these block the submission, not just the CV")
    sys.exit(1)
print("ready")
