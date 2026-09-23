"""E113 (23.09, hypothesis 25): airline prefix for rows without an NM record — lookup first, rules as fallback.
1) FLIGHT_mvt -> 3-letter prefix of CALLSIGN_flt, the mode over rows WITH an NM record (no target used; 99.99-100 %
   exact on held-out month pairs among NM rows, external check); it carries the operating callsign (DY -> NOZ, KM -> KMM)
   and, for registrations, the same first three letters the model already sees on NM rows.
2) E109 rules where the lookup has no entry: IATA -> ICAO via OpenFlights, doubled letters -> first three; registrations
   -> first three letters (not "_REG", which NM rows never carry).
    .venv/bin/python src/e113_prefix_lookup.py      data/features_e106.parquet -> data/features_e113.parquet
"""
import csv
import re
import polars as pl

iata = {}
for row in csv.reader(open("data/ext/openflights_airlines.dat", encoding="utf-8")):
    if len(row) > 7 and re.fullmatch(r"[A-Z0-9]{2}", row[3] or "") and re.fullmatch(r"[A-Z]{3}", row[4] or "") and row[7] == "Y":
        iata.setdefault(row[3], row[4])


def rules(fl):
    if fl is None:
        return None
    m = re.fullmatch(r"([A-Z0-9]{2})\d{1,4}[A-Z]?", fl)
    if m:
        return iata.get(m.group(1))
    m = re.fullmatch(r"([A-Z]{3})[A-Z]*\d.*", fl)
    if m:
        return m.group(1)
    m = re.fullmatch(r"([A-Z]{3})[A-Z0-9]{0,3}", fl.replace("-", ""))
    return m.group(1) if m else None


f = pl.read_parquet("data/features_e106.parquet")
nm = f.filter(pl.col("FLIGHT_ID_mvt").is_not_null() & pl.col("CALLSIGN_flt").is_not_null())
lk = (nm.with_columns(cp=pl.col("CALLSIGN_flt").str.extract(r"^([A-Z]{3})", 1)).drop_nulls(["FLIGHT_mvt", "cp"])
        .group_by("FLIGHT_mvt", "cp").len().sort("len", descending=True).group_by("FLIGHT_mvt").agg(_lk=pl.col("cp").first()))
todo = pl.col("FLIGHT_ID_mvt").is_null() & pl.col("airline_prefix").is_null()
before = f.filter(todo).height
f = f.join(lk, on="FLIGHT_mvt", how="left", maintain_order="left").with_columns(
    _r=pl.col("FLIGHT_mvt").map_elements(rules, return_dtype=pl.String))
f = f.with_columns(pl.when(todo).then(pl.coalesce("_lk", "_r")).otherwise(pl.col("airline_prefix")).alias("airline_prefix")).drop("_lk", "_r")
print("no-NM rows with empty prefix:", before, "->", f.filter(pl.col("FLIGHT_ID_mvt").is_null() & pl.col("airline_prefix").is_null()).height)
f.write_parquet("data/features_e113.parquet")
