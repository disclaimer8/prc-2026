"""How do ADES_mvt / ADES_flt / ADES_FILED_flt relate? Defines what 'diversion' means here."""
import polars as pl
from common import load_all
df = load_all().select("hub","PHASE_mvt","ADEP_mvt","ADES_mvt","ADEP_flt","ADES_flt","ADES_FILED_flt","FLIGHT_TYPE_flt","src").collect()
m = df.filter(pl.col("ADES_flt").is_not_null())
print("rows with flight info:", m.height, "of", df.height)
def cnt(expr, name):
    print(f"{name}: {m.filter(expr).height}")
cnt(pl.col("ADES_flt") != pl.col("ADES_FILED_flt"), "ADES_flt != ADES_FILED_flt")
cnt(pl.col("ADES_mvt") != pl.col("ADES_flt"), "ADES_mvt != ADES_flt")
cnt(pl.col("ADEP_mvt") != pl.col("ADEP_flt"), "ADEP_mvt != ADEP_flt")
d = m.filter(pl.col("ADES_flt") != pl.col("ADES_FILED_flt"))
print("\nDiverted by phase:")
print(d.group_by("PHASE_mvt").len())
print("ARR rows diverted: is hub == ADES_flt?", d.filter((pl.col("PHASE_mvt")=="ARR")&(pl.col("hub")==pl.col("ADES_flt"))).height,
      " hub == ADES_FILED?", d.filter((pl.col("PHASE_mvt")=="ARR")&(pl.col("hub")==pl.col("ADES_FILED_flt"))).height)
print("DEP rows diverted: return-to-origin (ADES_flt==ADEP_flt):", d.filter((pl.col("PHASE_mvt")=="DEP")&(pl.col("ADES_flt")==pl.col("ADEP_flt"))).height)
print("DEP rows diverted: ADES_mvt equals ADES_FILED (airport report still shows filed dest):", d.filter((pl.col("PHASE_mvt")=="DEP")&(pl.col("ADES_mvt")==pl.col("ADES_FILED_flt"))).height,
      " equals ADES_flt:", d.filter((pl.col("PHASE_mvt")=="DEP")&(pl.col("ADES_mvt")==pl.col("ADES_flt"))).height)
print("ARR diverted rows: ADES_FILED is one of hubs (i.e. hub-to-hub diversion):")
from common import HUBS
print(d.filter((pl.col("PHASE_mvt")=="ARR")&pl.col("ADES_FILED_flt").is_in(HUBS)).height)
with pl.Config(tbl_rows=30, tbl_cols=-1):
    print(d.sample(25, seed=1))
