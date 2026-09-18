"""Проверка вывода о длинном хвосте по СЫРЫМ трекам, а не по offblock2 (18.09).

    .venv/bin/python src/longtail_raw_probe.py [дней]

17.09 по `offblock2_*.parquet` вышло: у строк с рулением > 3600 с трек начинается на +4 091 с позже записанного
отхода, и только у 3 % он есть до записи отхода. Отсюда был сделан вывод «ответчик молчит, наблюдать нечего».
Но `t_first` в offblock2 — это начало **последнего непрерывного наземного отрезка**: проход назад в
`src/adsb_offblock_v2.py:38` обрывается на паузе длиннее GAP_S = 900 с. Рейс, простоявший час у стоянки, такую
паузу почти наверняка поймает, и тогда историю стоянки выбрасывает наш собственный код, а не отсутствие приёма.

Здесь то же самое считается по сырым точкам суток: берётся весь наземный трек того же борта до взлёта, без
проходов назад и без обрывов.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
from adsb_offblock_v2 import surface_mask  # noqa: E402

N_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 12


def main() -> None:
    f = pl.read_parquet("data/features_e28.parquet",
                        columns=["MVT_ID_mvt", "ADEP_mvt", "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt",
                                 "BLOCK_TIME_UTC_mvt", "CALLSIGN_flt", "FLIGHT_mvt", "split"])
    f = f.filter(pl.col("split") == "train").with_columns(
        day=pl.col("MVT_TIME_UTC_mvt").dt.date(),
        to=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64),
        block=pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64),
        cs=pl.col("CALLSIGN_flt").fill_null(pl.col("FLIGHT_mvt")).str.replace_all(" ", ""))
    days = (f.filter(pl.col("TAXITIME_SEC_mvt") > 3600).group_by("day").len()
             .sort("len", descending=True).head(N_DAYS)["day"].to_list())
    out = []
    for day in days:
        path = f"data/adsblol/surface_{day}.parquet"
        if not os.path.exists(path):
            print(f"{day}: файла нет", flush=True)
            continue
        pts = pl.read_parquet(path).with_columns(surf=surface_mask(pl.read_parquet(path)))
        o2 = pl.read_parquet(f"data/adsblol/offblock2_{day}.parquet", columns=["MVT_ID_mvt", "coverage", "t_first", "t_first_move"])
        dep = f.filter(pl.col("day") == day)
        cand: dict[tuple, list] = {}
        for cs, ap, icao in pts.filter(pl.col("callsign").is_not_null()).select("callsign", "airport", "icao").unique().iter_rows():
            cand.setdefault((cs, ap), []).append(icao)
        ground = {(k[0], k[1]): np.sort(np.asarray(v)) for k, v in
                  ((k, g["t"].to_numpy()) for k, g in pts.filter(pl.col("surf")).group_by(["icao", "airport"]))}
        d = dep.join(o2, on="MVT_ID_mvt", how="left")
        for mvt, ap, y, to, block, cs, cov, t_first, t_move in d.select(
                "MVT_ID_mvt", "ADEP_mvt", "TAXITIME_SEC_mvt", "to", "block", "cs", "coverage", "t_first", "t_first_move").iter_rows():
            ok = [i for i in cand.get((cs, ap), [])
                  if (ts := ground.get((i, ap))) is not None and len(b := ts[ts <= to + 300]) and to - b[-1] <= 900]
            if len(ok) != 1:
                continue
            ts = ground[(ok[0], ap)]
            before_to = ts[ts <= to + 300]
            raw_first = float(before_to[0])
            in_window = int(((ts >= block - 300) & (ts <= block + 300)).sum())
            span = ts[(ts >= block) & (ts <= to)]
            out.append((mvt, ap, float(y), raw_first - block, (t_first - block) if t_first is not None else None,
                        in_window, len(span), (float(np.max(np.diff(span))) if len(span) > 2 else None), cov))
        print(f"{day}: сопоставлено {len(out)}", flush=True)
    r = pl.DataFrame(out, schema=["MVT_ID_mvt", "ADEP_mvt", "y", "raw_first_minus_block", "seg_first_minus_block",
                                  "pts_around_block", "pts_block_to_to", "max_gap", "coverage"], orient="row")
    r.write_parquet("data/longtail_raw_probe.parquet")
    print()
    for lab, sub in (("длинные (y > 3600)", r.filter(pl.col("y") > 3600)),
                     ("обычные (600–1800)", r.filter(pl.col("y").is_between(600, 1800)))):
        n = sub.height
        if n == 0:
            continue
        raw = sub["raw_first_minus_block"].to_numpy()
        seg = sub["seg_first_minus_block"].drop_nulls().to_numpy()
        print(f"{lab}: n = {n}")
        print(f"   сырой трек: первая наземная точка минус запись отхода, медиана {np.median(raw):+8.0f} с; "
              f"есть точки ДО записи отхода у {np.mean(raw < 0) * 100:5.1f} %")
        if len(seg):
            print(f"   offblock2:  то же по отрезку,               медиана {np.median(seg):+8.0f} с; "
                  f"до записи отхода у {np.mean(seg < 0) * 100:5.1f} %")
        print(f"   точек в ±300 с от записи отхода: медиана {np.median(sub['pts_around_block'].to_numpy()):.0f}, "
              f"есть хоть одна у {np.mean(sub['pts_around_block'].to_numpy() > 0) * 100:.1f} %")
        print(f"   точек между записью отхода и взлётом: медиана {np.median(sub['pts_block_to_to'].to_numpy()):.0f}")
        g = sub["max_gap"].drop_nulls().to_numpy()
        if len(g):
            print(f"   наибольшая пауза в этом промежутке: медиана {np.median(g):.0f} с, "
                  f"длиннее 900 с у {np.mean(g > 900) * 100:.1f} %")
        print()


if __name__ == "__main__":
    main()
