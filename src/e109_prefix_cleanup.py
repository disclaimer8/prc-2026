"""E109 (23.09): finish the E106 airline prefix for rows without an NM record (no target used).
IATA-form flight numbers (KL1234) -> ICAO via OpenFlights (data/ext/openflights_airlines.dat, ODbL); doubled-letter
forms (WZZZZ401, KMMMM615) -> first three letters; registration-like strings (N501PC, OEGSF, PHTDS) -> "_REG".
    .venv/bin/python src/e109_prefix_cleanup.py      data/features_e106.parquet -> data/features_e109.parquet
"""
import csv
import re
import polars as pl

iata = {}
for row in csv.reader(open("data/ext/openflights_airlines.dat", encoding="utf-8")):
    if len(row) > 7 and re.fullmatch(r"[A-Z0-9]{2}", row[3] or "") and re.fullmatch(r"[A-Z]{3}", row[4] or "") and row[7] == "Y":
        iata.setdefault(row[3], row[4])


def fix(fl):
    if fl is None:
        return None
    m = re.fullmatch(r"([A-Z0-9]{2})\d{1,4}[A-Z]?", fl)
    if m:
        return iata.get(m.group(1))
    m = re.fullmatch(r"([A-Z]{3})[A-Z]*\d.*", fl)
    if m:
        return m.group(1)
    if re.fullmatch(r"(N\d{1,5}[A-Z]{0,2}|[A-Z0-9]{1,2}-?[A-Z0-9]{3,4})", fl):
        return "_REG"
    return None


f = pl.read_parquet("data/features_e106.parquet")
todo = pl.col("FLIGHT_ID_mvt").is_null() & pl.col("airline_prefix").is_null()
before = f.filter(todo).height
f = f.with_columns(pl.when(todo).then(pl.col("FLIGHT_mvt").map_elements(fix, return_dtype=pl.String))
                   .otherwise(pl.col("airline_prefix")).alias("airline_prefix"))
print("no-NM rows with empty prefix:", before, "->", f.filter(pl.col("FLIGHT_ID_mvt").is_null() & pl.col("airline_prefix").is_null()).height)
f.write_parquet("data/features_e109.parquet")
