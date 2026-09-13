"""Upload submissions/gentle-igloo_vN.parquet to the team bucket after checking it against the template."""
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

load_dotenv(".env")
client = Minio("s3.opensky-network.org", access_key=os.environ["PRC_ACCESS_KEY"], secret_key=os.environ["PRC_SECRET_KEY"], secure=True)
client.fput_object("prc-2026-gentle-igloo", name, f"submissions/{name}")
print("uploaded", name, [o.object_name for o in client.list_objects("prc-2026-gentle-igloo")])
