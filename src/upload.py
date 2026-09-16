"""Upload submissions/gentle-igloo_vN.parquet to the team bucket after checking it against the template and against our
current best submission (UPLOAD_REF, default v30; UPLOAD_MAX_RMS overrides the plausibility bound)."""
import os
import sys

import polars as pl
from dotenv import load_dotenv
from minio import Minio

version = int(sys.argv[1])
name = f"gentle-igloo_v{version}.parquet"
sub = pl.read_parquet(f"submissions/{name}")
template = pl.read_parquet("data/prc-2026-datasets/submitting.parquet")
assert sub.schema == template.schema, (sub.schema, template.schema)
assert sub["MVT_ID_mvt"].sort().equals(template["MVT_ID_mvt"].sort()), "MVT_ID set differs"
assert sub["TAXITIME_SEC_mvt"].null_count() == 0

# Guard against the most expensive failure of past editions: in 2025 eleven of thirty teams shipped a final file whose
# values were permuted relative to their own ids (a groupby reorder plus positional assignment), scoring 2-23x worse on rows
# they had already scored well. The format check passes such a file. So: compare against our previous best submission on the
# shared ids and refuse an implausible jump unless it is explicitly allowed.
ref = os.environ.get("UPLOAD_REF", "submissions/gentle-igloo_v30.parquet")  # current best
max_rms = float(os.environ.get("UPLOAD_MAX_RMS", "400"))
if os.path.exists(ref) and os.path.abspath(ref) != os.path.abspath(f"submissions/{name}"):
    prev = pl.read_parquet(ref)
    j = prev.join(sub, on="MVT_ID_mvt", suffix="_new")
    d = (j["TAXITIME_SEC_mvt_new"].cast(float) - j["TAXITIME_SEC_mvt"].cast(float)).to_numpy()
    rms = float((d ** 2).mean() ** 0.5)
    print(f"vs {ref}: {len(d)} shared rows, RMS change {rms:.1f} s, changed {(d != 0).sum()}")
    assert rms <= max_rms, f"RMS change {rms:.0f} s vs {ref} exceeds {max_rms:.0f} s — looks like a broken assembly, not a model change (set UPLOAD_MAX_RMS to override)"

load_dotenv(".env")
client = Minio("s3.opensky-network.org", access_key=os.environ["PRC_ACCESS_KEY"], secret_key=os.environ["PRC_SECRET_KEY"], secure=True)
client.fput_object("prc-2026-gentle-igloo", name, f"submissions/{name}")
print("uploaded", name, [o.object_name for o in client.list_objects("prc-2026-gentle-igloo")])
