"""Признаки предыстории стоянки для ADS-B-этапа (18.09).

    .venv/bin/python src/stand_history.py <день>      пишет data/adsblol/hist_<день>.parquet

Что это. `src/adsb_offblock_v2.py` строит наземный отрезок проходом назад, который обрывается на паузе длиннее
GAP_S = 900 с. Всё, что борт делал на земле до этого обрыва, в признаки этапа не попадает вовсе. Замер 18.09 по
сырым суткам: у строк с рулением больше часа такая пауза есть в 38,8 % случаев против 0,3 % у обычных, наземные
точки до записи отхода есть у 76,2 % длинных строк, а между записью отхода и взлётом их медианно 72.

Почему признаки, а не продление отрезка. Продление обязано решить, где началось руление, и потому не может
перешагнуть движение — иначе проглатывает руление прилёта (E61: правка добавила 16 % оценок, но до длинного
хвоста не достала, +21 строка из 3 211). Признаки ничего не решают: они лишь описывают, что борт делал на земле
раньше, и потому безопасны там, где продление опасно.

Считается по всем наземным точкам того же борта в том же аэропорту ДО начала отрезка offblock2:
    hist_seen_s      от первой наземной точки предыстории до начала отрезка
    hist_stat_s      сколько из этого борт стоял (gs <= MOVING_KT)
    hist_gaps        число пауз длиннее GAP_S в предыстории
    hist_max_gap     наибольшая пауза, секунды
    hist_moved_m     расстояние от первой точки предыстории до начала отрезка
    hist_last_move_s от последнего движения в предыстории до начала отрезка (null, если движения не было)
Строки без предыстории получают нули по количествам и null по временам последнего движения — то есть «предыстории
нет» отличимо от «предыстория есть и в ней не двигались».
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
import adsb_offblock_v2 as V  # noqa: E402

COLS = ["MVT_ID_mvt", "hist_seen_s", "hist_stat_s", "hist_gaps", "hist_max_gap", "hist_moved_m", "hist_last_move_s"]
R = 6371000.0


def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(min(max(a, 0.0), 1.0))))


def main() -> None:
    day = sys.argv[1]
    pts = pl.read_parquet(f"data/adsblol/surface_{day}.parquet")
    pts = pts.with_columns(surf=V.surface_mask(pts)).filter(pl.col("surf")).sort("icao", "t")
    o2 = pl.read_parquet(f"data/adsblol/offblock2_{day}.parquet", columns=["MVT_ID_mvt", "coverage", "t_first", "t_to"])
    y, m, d = map(int, day.split("-"))
    src = pl.read_parquet("data/prc-2026-datasets/" + ("training_*.parquet" if y == 2025 else "ranking.parquet"))
    dep = src.filter((pl.col("PHASE_mvt") == "DEP") & (pl.col("MVT_TIME_UTC_mvt").dt.date() == pl.date(y, m, d))).with_columns(
        cs=pl.col("CALLSIGN_flt").fill_null(pl.col("FLIGHT_mvt")).str.replace_all(" ", ""),
        to=pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64)).join(o2, on="MVT_ID_mvt", how="inner")
    cand: dict[tuple, list] = {}
    for cs, ap, icao in pts.filter(pl.col("callsign").is_not_null()).select("callsign", "airport", "icao").unique().iter_rows():
        cand.setdefault((cs, ap), []).append(icao)
    by = {(k[0], k[1]): g for k, g in pts.group_by(["icao", "airport"])}
    out = []
    for mvt, ap, cs, to, t_first in dep.select("MVT_ID_mvt", "ADEP_mvt", "cs", "to", "t_first").iter_rows():
        if t_first is None:
            continue
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
        t = g["t"].to_numpy()
        pre = t < t_first
        if pre.sum() == 0:
            out.append((mvt, 0.0, 0.0, 0, 0.0, 0.0, None))
            continue
        tp = t[pre]
        gsp = np.nan_to_num(g["gs"].to_numpy()[pre])
        latp, lonp = g["lat"].to_numpy()[pre], g["lon"].to_numpy()[pre]
        i_first = int(np.argmax(t >= t_first))
        dt = np.diff(tp, prepend=tp[0])
        gaps = dt[1:] if len(dt) > 1 else np.array([0.0])
        moving = gsp > V.MOVING_KT
        last_move = float(t_first - tp[moving][-1]) if moving.any() else None
        out.append((mvt, float(t_first - tp[0]), float(dt[~moving].sum()), int((gaps > V.GAP_S).sum()),
                    float(gaps.max() if len(gaps) else 0.0),
                    hav(latp[0], lonp[0], g["lat"].to_numpy()[i_first], g["lon"].to_numpy()[i_first]), last_move))
    pl.DataFrame(out, schema=COLS, orient="row").write_parquet(f"data/adsblol/hist_{day}.parquet")
    print(day, len(out), flush=True)


if __name__ == "__main__":
    main()
