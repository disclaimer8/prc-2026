"""E88: здоровье приёма аэропорта как признак — из уже скачанных треков, без переизвлечения (21.09).

    .venv/bin/python src/e88_reception_health.py    data/features_e77.parquet -> data/features_e88.parquet

Предложение внешнего разбора: строить совместную модель состояния борта и процесса наблюдения, где главный
новый сигнал — «пропал ли приём у соседних бортов одновременно». Одновременный провал говорит о приёмнике,
исчезновение одного борта при живом приёме остальных — о самолёте. Ни одной колонки об этом у нас нет.

Диагностика 21.09 на 12 сутках (по одним 15-м числам) измерила потолок и границы:

* среди **частично видимых** рейсов (покрытие 1–2) худшая четверть по приёму даёт среднюю |ошибку| 109,7 с
  против ~85 у остальных трёх четвертей; это 8 715 строк выборки и 21,5 % её SSE, а полный съём их избытка
  стоил бы **7,24 с** в единицах ворот;
* связь **немонотонна** — весь эффект в худшей четверти, дальше плато, то есть это режим сорванного приёма,
  а не непрерывная зависимость;
* там, где покрытия нет вовсе, связи **нет** (147 / 138 / 135 / 136 по квартилям): LTFM эта линия не лечит;
* первая, наивная версия замера показывала сильную связь, которая целиком оказалась **трафиком** — число
  видимых бортов ночью мало не из-за приёмника. Поэтому мера нормирована на трафик: точек на борт.

Три колонки, все на уровне аэропорта и пятиминутки, агрегированные по окну [взлёт − 30 мин, взлёт]:

    rx_dens   средняя плотность приёма (точек на видимый борт), делённая на медиану своего аэропорта
    rx_drop   наибольшая доля бортов, видимых в предыдущей пятиминутке и пропавших в текущей
    rx_blank  число пятиминуток окна, где аэропорт не виден вовсе

Почему это правильный первый шаг, а не модель последовательности: большая модель питается тем же сигналом.
Если три колонки ворот не проходят, строить её незачем; если проходят — тогда и обсуждать состояния, уже
зная, что информация есть.

ЗАПИСАНО ДО ПРОГОНА. Ворота стандартные: решаемые строки, 6 фолдов, ≥ 0,3 с, ниже в ≥ 4 из 6, то же на
y ≤ 7200 и на усечённом. Прогноз: **−0,05 … −0,60 с, центр −0,15**. Центр мал из-за немонотонности и плато:
снять удастся малую долю тех 7,24 с, и только на частично видимых строках.
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import polars as pl

BIN = 300
WINDOW_BINS = 6  # 30 минут до взлёта


def health_table() -> pl.DataFrame:
    parts = []
    for p in sorted(glob.glob("data/adsblol/surface_*.parquet")):
        d = pl.read_parquet(p, columns=["icao", "airport", "t", "on_ground"]).filter(pl.col("on_ground"))
        if d.height == 0:
            continue
        g = (d.with_columns(bin=(pl.col("t") // BIN).cast(pl.Int64))
             .group_by("airport", "bin")
             .agg(live=pl.col("icao").n_unique(), pts=pl.len(), ics=pl.col("icao").unique()))
        parts.append(g)
    h = pl.concat(parts).sort("airport", "bin")
    h = h.with_columns(dens=pl.col("pts") / pl.col("live"))
    # доля бортов предыдущей пятиминутки, пропавших в текущей
    h = h.with_columns(prev=pl.col("ics").shift(1).over("airport"), prev_bin=pl.col("bin").shift(1).over("airport"))
    h = h.with_columns(
        drop=pl.when(pl.col("prev_bin") == pl.col("bin") - 1)
        .then(1.0 - (pl.col("prev").list.set_intersection(pl.col("ics")).list.len() / pl.col("prev").list.len()))
        .otherwise(None))
    return h.select("airport", "bin", "live", "dens", "drop")


def main() -> None:
    src = os.environ.get("FEATURES", "data/features_e77.parquet")
    dst = os.environ.get("FEATURES_OUT", "data/features_e88.parquet")
    h = health_table()
    med = h.group_by("airport").agg(m=pl.col("dens").median())
    h = h.join(med, on="airport").with_columns(dens_r=pl.col("dens") / pl.col("m"))
    print(f"пятиминуток приёма: {h.height:,}")

    df = pl.read_parquet(src)
    base = df.select("MVT_ID_mvt", "ADEP_mvt", t_bin=(pl.col("t_mvt") // BIN).cast(pl.Int64))
    acc = []
    for k in range(WINDOW_BINS):
        acc.append(base.with_columns(bin=pl.col("t_bin") - k)
                   .join(h.select("airport", "bin", "dens_r", "drop"),
                         left_on=["ADEP_mvt", "bin"], right_on=["airport", "bin"], how="left")
                   .select("MVT_ID_mvt", "dens_r", "drop"))
    w = pl.concat(acc).group_by("MVT_ID_mvt").agg(
        rx_dens=pl.col("dens_r").mean(),
        rx_drop=pl.col("drop").max(),
        rx_blank=pl.col("dens_r").is_null().sum().cast(pl.Int8))
    res = df.join(w, on="MVT_ID_mvt", how="left")

    new = [c for c in res.columns if c not in df.columns]
    if len(res) != len(df) or [c for c in df.columns if c not in res.columns]:
        sys.exit("отказ: результат не является надстройкой над исходником")
    for c in df.columns:
        if not df[c].equals(res[c]):
            sys.exit(f"отказ: изменилась исходная колонка {c}")
    res.write_parquet(dst)
    print(f"{src} -> {dst}: {res.height} строк, {len(new)} новых колонок: {', '.join(new)}")
    for c in new:
        nn = res[c].null_count()
        print(f"  {c}: пропусков {100 * nn / res.height:.1f} %, медиана {res[c].median()}")


if __name__ == "__main__":
    main()
