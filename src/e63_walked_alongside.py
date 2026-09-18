"""E63: продлённый отрезок ДОБАВЛЕН рядом, а не подставлен вместо (18.09).

    .venv/bin/python src/e63_walked_alongside.py

Что проверяется. E61 подменил `t_first`/`t_first_move` значениями с починенного прохода назад: +0,25 с, хуже во
всех шести фолдах. E62 сделал то же честно — не переклеивал класс покрытия и дал три признака-индикатора: +0,22 с,
то есть вернул 0,03 из 0,25. Гипотеза «модель ломает незнание, что значение восстановлено» опровергнута.

Оставшееся объяснение: модель ломает **потеря**. У 186 705 строк `t_first` уехал назад, и признак
`mvt_first = взлёт − первая наземная точка` сменил смысл — раньше это была длина наблюдённого куска, теперь длина
куска плюс сколько его удалось дотянуть. Индикатор сообщает, что величина другая, но прежней не возвращает.
Разложение E61 по решаемым строкам: вред этих 186 705 строк +0,146 с против +0,118 с у 1 787 063 нетронутых, то
есть в двенадцать раз больше на строку.

Здесь ничего не подменяется: обе стороны считаются на **контрольном** извлечении, а продлённый проход приходит
пятью отдельными колонками (`sw_mvt_first`, `sw_mvt_first_move`, `sw_coverage`, `sw_extra_s`, `sw_recovered`).
Единственное различие плеч — список признаков этапа.

Ожидание, записанное ДО прогона. Выигрыш ограничен строками с новой информацией: их 68 273, они держат 0,79 с
ошибки контроля, и E61 снял с них 0,063 с. Значит потолок здесь примерно −0,06…−0,1 с при пороге −0,3 —
**гейт этот эксперимент почти наверняка не возьмёт**, и ставится он ради механизма, а не ради лидерборда:
  * если вред исчезнет и итог станет около −0,06 — подтверждается, что ломала именно потеря прежней величины, и
    правило «менять вход только добавлением колонки» идёт в бриф;
  * если вред останется — значит дело не в подаче вообще, и класс «дать этапу больше про наземный трек»
    закрывается тремя независимыми прогонами, а не двумя.

Гейт: SSE по строкам, которые решает базовая модель, ниже контроля на >= 0,3 с LB-эквивалента и ниже в >= 4 из 6,
и то же на строках y <= 7200. Strict, полная и обрезанная показываются, но не гейтятся.
"""
import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

sys.path.insert(0, os.path.dirname(__file__))
_argv, sys.argv = sys.argv, [sys.argv[0], "import"]
import e36_adsb_stand_geometry as E36  # noqa: E402
sys.argv = _argv
from fold_metrics import compare  # noqa: E402

CTRL_DIR, CAND_DIR = "data/cv_preds_e63_ctrl", "data/cv_preds_e63"
SW = ["sw_mvt_first", "sw_mvt_first_move", "sw_coverage", "sw_extra_s", "sw_recovered"]

raw = pl.concat([pl.read_parquet(f, columns=E36.cols) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(
    pl.col("PHASE_mvt") == "DEP")
E36.ADSB_DIR = "data/adsblol"
d = E36.build(raw, E36.oof_base(), E36.load_adsb("data/adsblol/offblock2_2025-*.parquet")).with_columns(
    month=pl.col("MVT_TIME_UTC_mvt").dt.month())

walked = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "coverage", "t_first", "t_first_move"])
                    for f in sorted(glob.glob("data/adsblol_standwalk/offblock2_2025-*.parquet"))]).rename(
    {"coverage": "sw_coverage", "t_first": "t_first_sw", "t_first_move": "t_first_move_sw"})
d = d.join(walked, on="MVT_ID_mvt", how="left").with_columns(
    sw_mvt_first=pl.col("to") - pl.col("t_first_sw"),
    sw_mvt_first_move=pl.col("to") - pl.col("t_first_move_sw"),
    sw_extra_s=pl.col("t_first") - pl.col("t_first_sw"),
).with_columns(sw_recovered=(pl.col("sw_extra_s").fill_null(0) > 0).cast(pl.Int8))
print(f"строк {d.height}; продлённый отрезок известен у {int(d['sw_mvt_first'].is_not_null().sum())}, "
      f"продлён у {int(d['sw_recovered'].sum())}, первое движение появилось у "
      f"{int((d['mvt_first_move'].is_null() & d['sw_mvt_first_move'].is_not_null()).sum())}", flush=True)

for p in (CTRL_DIR, CAND_DIR):
    os.makedirs(p, exist_ok=True)
for a, b in [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]:
    tr, te = d.filter(~pl.col("month").is_in([a, b])), d.filter(pl.col("month").is_in([a, b]))
    y = te[E36.Y].to_numpy()
    line = []
    for name, out, feats in (("ctrl", CTRL_DIR, E36.FEATS), ("cand", CAND_DIR, E36.FEATS + SW)):
        pred = E36.correct(E36.fit_stage(tr, feats), te)
        np.savez(f"{out}/fold_{a}_{b}.npz", mvt_id=te["MVT_ID_mvt"].to_numpy(), pred=pred, y=y)
        line.append(f"{name} RMSE {np.sqrt(np.mean((pred - y) ** 2)):.2f}")
    print(f"fold {a}+{b}: " + " | ".join(line), flush=True)

dec, tot, small, trim, lower, n, strict = compare(CTRL_DIR, CAND_DIR)
ok = n == 6 and -dec >= 0.3 and lower >= 4 and -small >= 0.3
print(f"DECISION E63: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s)")
