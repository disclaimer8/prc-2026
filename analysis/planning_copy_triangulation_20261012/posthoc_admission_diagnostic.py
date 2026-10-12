"""Post-hoc technical admission diagnostic; NO y/BLOCK/planning/flag/effect columns.

Invoked after frozen #464 returned zero raw stationary prefixes. No thresholds,
samples, original outputs or source are modified. Only first-point gates/types.
"""
from pathlib import Path
import importlib.util
import json
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REG = "bf08d014000bd51ad65ae045be644ab0629f6c8f"


def imported(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def main():
    t0 = time.monotonic()
    frozen = imported("frozen_tri", ROOT/"scripts/planning_copy_triangulation_20261012.py")
    import numpy as np
    import polars as pl
    manifest = json.loads((OUT/"preflight.json").read_text())
    frozen.verify_registration(ROOT, REG, manifest)
    geo = imported("frozen_geo", ROOT/"src/geometry.py")
    geo.OSM_DIR = ROOT/"analysis/sources/samples/osm_overpass"
    st, _ = geo.load_tables()
    exact = {(a,k): (la,lo) for a,k,la,lo in st.iter_rows()}
    suffix = {}
    for a,k,la,lo in st.iter_rows():
        suffix.setdefault((a,geo.strip_suffix(k)),(la,lo))
    # Only IDs/calendar/airport from prior audit; do not read its target-related columns.
    ids = pl.read_csv(OUT/"raw_episode_audit.csv", columns=["MVT_ID_mvt","day","airport"])
    cols = ["MVT_ID_mvt","PHASE_mvt","MVT_TIME_UTC_mvt","ADEP_mvt","CALLSIGN_flt","FLIGHT_mvt","STAND_mvt"]
    dep = pl.concat([pl.scan_parquet(ROOT/f["path"]).select(cols) for f in manifest["files"] if f["kind"]=="training"])
    dep = dep.filter(pl.col("PHASE_mvt")=="DEP").with_columns(pl.col("MVT_ID_mvt").cast(pl.Int64)).join(ids.lazy(),on="MVT_ID_mvt",how="inner",validate="1:1").collect()
    dep = dep.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.epoch("us").cast(pl.Float64).truediv(1e6).alias("takeoff"),
                           pl.col("CALLSIGN_flt").fill_null(pl.col("FLIGHT_mvt")).str.replace_all(" ","").alias("cs"))
    rows, days = [], []
    for file in [f for f in manifest["files"] if f["kind"]=="raw"]:
        day = file["day"]
        visits = dep.filter(pl.col("day")==day)
        if not file["stale_available"]:
            days.append(dict(day=day, rows=visits.height, stale_schema=False, diagnostic_rows=0))
            continue
        pts = pl.read_parquet(file["path"],columns=frozen.RAW_COLS).filter(pl.col("airport").is_in(frozen.AIRPORTS))
        elev = pl.col("airport").replace_strict(frozen.ELEV_FT,return_dtype=pl.Float64)
        pts = pts.with_columns((pl.col("on_ground") | ((pl.col("gs").fill_null(0)<=40)&(pl.col("alt").is_null()|(pl.col("alt")<=elev+800)))).fill_null(False).alias("surf"),
                               pl.col("stale").is_null().alias("stale_was_null"), pl.col("stale").fill_null(True),
                               pl.col("callsign").str.replace_all(" ","")).sort("airport","icao","t")
        candidates = {}
        for cs,ap,icao in pts.filter(pl.col("callsign").is_not_null()).select("callsign","airport","icao").unique().iter_rows():
            candidates.setdefault((cs,ap),[]).append(icao)
        tracks = {(ap,icao):{k:g[k].to_numpy() for k in ["t","lat","lon","gs","surf","stale","stale_was_null"]}
                  for (ap,icao),g in pts.group_by("airport","icao")}
        before = len(rows)
        for r in visits.iter_rows(named=True):
            key = geo.norm_stand(r["STAND_mvt"]) if r["STAND_mvt"] else None
            stand = exact.get((r["airport"],key)) or suffix.get((r["airport"],geo.strip_suffix(key))) if key else None
            options=[]
            for icao in candidates.get((r["cs"],r["airport"]),[]):
                tr=tracks[(r["airport"],icao)]
                ts=tr["t"][tr["surf"]&(tr["t"]<=r["takeoff"]+300)]
                if len(ts) and r["takeoff"]-ts[-1]<=900: options.append(icao)
            if len(options)!=1 or stand is None:
                raise RuntimeError("diagnostic association diverges from original available-stale path")
            tr=tracks[(r["airport"],options[0])]
            hit=np.flatnonzero(tr["surf"]&(tr["t"]<=r["takeoff"]+300))
            end=int(hit[-1]); start=end
            while start>0 and tr["surf"][start-1] and 0<=tr["t"][start]-tr["t"][start-1]<=900: start-=1
            dist=float(frozen.local_distance(tr["lat"][start],tr["lon"][start],*stand))
            displacement=float(frozen.local_distance(tr["lat"][start],tr["lon"][start],tr["lat"][start],tr["lon"][start]))
            gates=dict(surface=bool(tr["surf"][start]),speed_finite=bool(np.isfinite(tr["gs"][start])),
                       speed_0_1=bool(0<=tr["gs"][start]<=1),fresh=not bool(tr["stale"][start]),
                       near_stand=bool(np.isfinite(dist) and dist<=40),displacement_25=bool(displacement<=25),
                       before_takeoff=bool(tr["t"][start]<=r["takeoff"]))
            check=frozen.raw_stationary_prefix(tr,r["takeoff"],stand)
            rows.append(dict(MVT_ID_mvt=r["MVT_ID_mvt"],day=day,airport=r["airport"],**gates,
                        first_failing_gate=next((k for k,v in gates.items() if not v),"none"),
                        stale_was_null=bool(tr["stale_was_null"][start]),stale_filled=bool(tr["stale"][start]),
                        negation_matches_logical_not=bool((~tr["stale"])[start])==(not bool(tr["stale"][start])),
                        stale_numpy_dtype=str(tr["stale"].dtype),gs_numpy_dtype=str(tr["gs"].dtype),t_numpy_dtype=str(tr["t"].dtype),
                        raw_t=float(tr["t"][start]),takeoff=float(r["takeoff"]),stand_distance_m=dist,
                        gs=float(tr["gs"][start]) if np.isfinite(tr["gs"][start]) else None,
                        first_before_takeoff_s=float(r["takeoff"]-tr["t"][start]),
                        reproduced_n_stationary=check["n_stationary"]))
        days.append(dict(day=day,rows=visits.height,stale_schema=True,diagnostic_rows=len(rows)-before))
    d=pl.DataFrame(rows)
    d.write_csv(OUT/"posthoc_admission_rows.csv")
    report=dict(status="posthoc_technical_diagnostic_not_new_scientific_analysis",registration=REG,
                command=".venv/bin/python analysis/planning_copy_triangulation_20261012/posthoc_admission_diagnostic.py",
                script_sha256=frozen.sha(Path(__file__)),input_manifest_sha256=frozen.sha(OUT/"preflight.json"),
                id_pool_sha256=frozen.sha(OUT/"raw_episode_audit.csv"),fields_read=cols,
                prohibited_fields_read=[],rows=d.height,days=days,
                marginal_failures={k:int((~d[k]).sum()) for k in gates},
                first_failures=d.group_by("first_failing_gate").agg(pl.len().alias("n")).to_dicts(),
                stale_null_first=int(d["stale_was_null"].sum()),stale_true_first=int(d["stale_filled"].sum()),
                negation_mismatches=int((~d["negation_matches_logical_not"]).sum()),
                dtypes=d.select("stale_numpy_dtype","gs_numpy_dtype","t_numpy_dtype").unique().to_dicts(),
                max_reproduced_n=int(d["reproduced_n_stationary"].max()),
                time_ranges=d.select(pl.min("raw_t").alias("raw_min"),pl.max("raw_t").alias("raw_max"),
                                     pl.min("takeoff").alias("takeoff_min"),pl.max("takeoff").alias("takeoff_max")).to_dicts(),
                elapsed_seconds=time.monotonic()-t0,rows_sha256=frozen.sha(OUT/"posthoc_admission_rows.csv"))
    (OUT/"POSTHOC_ADMISSION_DIAGNOSTIC.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("days",)},indent=2))


if __name__=="__main__": main()
