"""Download PRC Data Challenge 2026 files from the OpenSky MinIO endpoint.

Credentials come from .env (PRC_ACCESS_KEY / PRC_SECRET_KEY). Every readable
bucket is listed; objects are mirrored into data/<bucket>/ and skipped when a
local file of the same size already exists.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from minio import Minio

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

client = Minio(
    "s3.opensky-network.org",
    access_key=os.environ["PRC_ACCESS_KEY"],
    secret_key=os.environ["PRC_SECRET_KEY"],
    secure=True,
)

for bucket in client.list_buckets():
    print(f"bucket {bucket.name}")
    try:
        objects = list(client.list_objects(bucket.name, recursive=True))
    except Exception as exc:  # submission buckets may be write-only
        print(f"  cannot list: {exc}")
        continue
    for obj in objects:
        dest = ROOT / "data" / bucket.name / obj.object_name
        if dest.exists() and dest.stat().st_size == obj.size:
            print(f"  have {obj.object_name}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  get  {obj.object_name} ({obj.size / 1e6:.1f} MB)")
        client.fget_object(bucket.name, obj.object_name, str(dest))
