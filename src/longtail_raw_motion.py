"""Вторая половина проверки (18.09): видно ли по СЫРОМУ треку само движение около записи отхода.

    .venv/bin/python src/longtail_raw_motion.py [дней]

Первая половина (`src/longtail_raw_probe.py`) показала, что борт принимается и до записи отхода, и между ней и
взлётом — «тишины» нет, её создавал обрыв прохода назад. Остался вопрос, от которого зависит тип возможного
изменения: **двигался ли борт в момент записанного отхода?**

* если двигался — метка физическая: борт отъехал от стоянки и потом стоял, и это наблюдаемое событие (второй,
  переносящийся тип);
* если стоял — запись отхода административная, и вчерашний вывод «метка совпадает с расписанием» остаётся,
  просто обоснован он был неверно.

Считается по всем наземным точкам борта, без проходов назад: скорость около записи отхода, смещение за первые
десять минут после неё, первое движение после неё и ошибка наблюдённого руления «первое движение → взлёт».
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
import adsb_offblock_v2 as V  # noqa: E402

N_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
R = 6371000.0


def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def main() -> None:
    f = pl.read_parquet("data/features_e28.parquet",
                        columns=["MVT_ID_mvt", "ADEP_mvt", "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt",
                                 "BLOCK_TIME_UTC_mvt", "CALLSIGN_flt", "FLIGHT_mvt", "split", "d_mvt_sched"])
    f = f.filter(pl.col("split") == "train").with_columns(
        day=pl.col("MVT_TIME_UTC_mvt").dt.date(),
        to=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64),
        block=pl.col("BLOCK_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64),
        cs=pl.col("CALLSIGN_flt").fill_null(pl.col("FLIGHT_mvt")).str.replace_all(" ", ""))
    days = (f.filter(pl.col("TAXITIME_SEC_mvt") > 3600).group_by("day").len()
             .sort("len", descending=True).head(N_DAYS)["day"].to_list())
    rows = []
    for day in days:
        path = f"data/adsblol/surface_{day}.parquet"
        if not os.path.exists(path):
            continue
        pts = pl.read_parquet(path)
        pts = pts.with_columns(surf=V.surface_mask(pts)).filter(pl.col("surf")).sort("icao", "t")
        cand: dict[tuple, list] = {}
        for cs, ap, icao in pts.filter(pl.col("callsign").is_not_null()).select("callsign", "airport", "icao").unique().iter_rows():
            cand.setdefault((cs, ap), []).append(icao)
        by = {(k[0], k[1]): g for k, g in pts.group_by(["icao", "airport"])}
        dep = f.filter(pl.col("day") == day)
        for mvt, ap, y, to, block, cs, dsched in dep.select(
                "MVT_ID_mvt", "ADEP_mvt", "TAXITIME_SEC_mvt", "to", "block", "cs", "d_mvt_sched").iter_rows():
            ok = []
            for i in cand.get((cs, ap), []):
                g = by.get((i, ap))
                if g is None:
                    continue
                t = g["t"].to_numpy()
                b = t[t <= to + 300]
                if len(b) and to - b[-1] <= 900:
                    ok.append(i)
            if len(ok) != 1:
                continue
            g = by[(ok[0], ap)]
            t, gs, lat, lon = g["t"].to_numpy(), np.nan_to_num(g["gs"].to_numpy()), g["lat"].to_numpy(), g["lon"].to_numpy()
            near = (t >= block - 300) & (t <= block + 300)
            after = (t >= block) & (t <= to)
            if after.sum() < 3:
                continue
            gs_near = float(gs[near].max()) if near.any() else None
            i0 = int(np.argmax(t >= block))
            far10 = (t >= block) & (t <= block + 600)
            disp10 = float(hav(lat[i0], lon[i0], lat[far10][-1], lon[far10][-1])) if far10.sum() > 1 else None
            mov = after & (gs > V.MOVING_KT)
            t_move = float(t[mov][0]) if mov.any() else None
            rows.append((mvt, ap, float(y), float(dsched) if dsched is not None else None, gs_near, disp10,
                         (t_move - block) if t_move is not None else None,
                         (to - t_move) if t_move is not None else None))
        print(f"{day}: {len(rows)}", flush=True)
    r = pl.DataFrame(rows, schema=["MVT_ID_mvt", "ADEP_mvt", "y", "d_mvt_sched", "gs_near_block", "disp_10min",
                                   "move_minus_block", "raw_observed_taxi"], orient="row")
    r.write_parquet("data/longtail_raw_motion.parquet")
    print()
    for lab, sub in (("длинные (y > 3600)", r.filter(pl.col("y") > 3600)),
                     ("обычные (600–1800)", r.filter(pl.col("y").is_between(600, 1800)))):
        if sub.height == 0:
            continue
        print(f"{lab}: n = {sub.height}")
        gsn = sub["gs_near_block"].drop_nulls().to_numpy()
        print(f"   скорость в ±300 с от записи отхода: медиана максимума {np.median(gsn):.1f} уз; "
              f"выше 3 уз у {np.mean(gsn > V.MOVING_KT) * 100:.1f} % (n={len(gsn)})")
        d10 = sub["disp_10min"].drop_nulls().to_numpy()
        print(f"   смещение за 10 мин после записи отхода: медиана {np.median(d10):.0f} м; "
              f"больше 150 м у {np.mean(d10 > 150) * 100:.1f} %")
        mm = sub["move_minus_block"].drop_nulls().to_numpy()
        print(f"   первое движение после записи отхода: медиана {np.median(mm):+.0f} с")
        obs = sub.drop_nulls("raw_observed_taxi")
        e = (obs["raw_observed_taxi"] - obs["y"]).to_numpy()
        print(f"   наблюдённое руление (первое движение -> взлёт) против факта: медиана {np.median(e):+.0f} с, "
              f"RMSE {np.sqrt((e ** 2).mean()):.0f} с, |ошибка| <= 300 с у {np.mean(np.abs(e) <= 300) * 100:.1f} %")
        print()


if __name__ == "__main__":
    main()
