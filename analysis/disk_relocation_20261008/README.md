# Relocated from the Mac on 2026-10-08 ~14:45Z (disk at 140 MB free)

`data/adsblol/surface_2025-*.parquet` (364 files, 4.72 GB) were removed from the Mac after verifying that byte-identical
copies exist on Home (`ssh home-pc-lan`, `/home/a1/prc-nnfinal-20261008/surface_mac/`): SHA-256 of all 364 files matched
(`surface_2025_sha256.txt`). The final tail-assembly contract does not bind these files; the 2025 derived stage inputs
(`offblock*`, `visit*`, `head`, `stale`) and all 2026 surface files remain on the Mac. To restore:
`scp 'home-pc-lan:/home/a1/prc-nnfinal-20261008/surface_mac/surface_2025-*' data/adsblol/` and check against this manifest.
