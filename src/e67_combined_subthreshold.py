"""E67: два подпороговых, но устойчивых изменения вместе (18.09).

    .venv/bin/python src/e67_combined_subthreshold.py

Два эксперимента не прошли гейт −0,3, но оба оказались ниже контроля **во всех шести фолдах**, то есть их знак
не случаен:
  * **E63** — продлённый проход назад по наземному треку, добавленный пятью отдельными колонками: −0,05 с, 6/6;
  * **E66** — где именно на маршруте пойман борт и сколько осталось до порога ВПП: −0,15 с, 6/6.
Вместе их никто не мерил. Контроль здесь — **принятая** конфигурация E65 (`E36.FEATS + adsb_visit.NAMES`,
−1,73 с, на лидерборде 248,7737), то есть меряется прирост к тому, что уже стоит в сабмите.

Подача — та же, что у E63, и это принципиально: обе стороны считаются на **контрольном** извлечении
`data/adsblol`, продлённый проход приходит пятью отдельными колонками, ни одно существующее значение не
подменяется. Иначе в один прогон попали бы два изменения сразу (E62-извлечение заодно двигает `t_first_move`),
и число было бы нечитаемым. Три колонки E66 уже лежат в `offblock5` и берутся оттуда как есть.

Ожидание, записанное ДО прогона: **около −0,20 с, гейт не проходит**. Складываться полностью они не обязаны:
E63 работает на строках, где отрезок продлился, E66 — на строках с известной геометрией, популяции пересекаются.
Интересен ровно один исход: если выйдет заметно больше −0,3, значит вместе они дают больше суммы, и тогда это
кандидат на сегодняшние слоты. Если выйдет около −0,2, направление закрывается двумя числами вместо одного.

Гейт: по решающей метрике ниже контроля на >= 0,3 с и ниже в >= 4 из 6, то же на y <= 7200, и выигрыш должен
пережить выбрасывание трёх лучших и трёх худших строк фолда.
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
from adsb_visit import NAMES  # noqa: E402
from fold_metrics import compare  # noqa: E402

CTRL_DIR, CAND_DIR = "data/cv_preds_e67_ctrl", "data/cv_preds_e67"
SW = ["sw_mvt_first", "sw_mvt_first_move", "sw_coverage", "sw_extra_s", "sw_recovered"]
NEW = ["gs_d_first_rwy", "gs_frac_done", "gs_d_stand_rwy"]

raw = pl.concat([pl.read_parquet(f, columns=E36.cols) for f in sorted(glob.glob("data/prc-2026-datasets/training_*.parquet"))]).filter(
    pl.col("PHASE_mvt") == "DEP")
vis = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob("data/adsblol/visit_2025-*.parquet"))])
E36.ADSB_DIR = "data/adsblol"
d = E36.build(raw, E36.oof_base(), E36.load_adsb("data/adsblol/offblock2_2025-*.parquet")).join(
    vis, on="MVT_ID_mvt", how="left").with_columns(month=pl.col("MVT_TIME_UTC_mvt").dt.month())

walked = pl.concat([pl.read_parquet(f, columns=["MVT_ID_mvt", "coverage", "t_first", "t_first_move"])
                    for f in sorted(glob.glob("data/adsblol_standwalk/offblock2_2025-*.parquet"))]).rename(
    {"coverage": "sw_coverage", "t_first": "t_first_sw", "t_first_move": "t_first_move_sw"})
d = d.join(walked, on="MVT_ID_mvt", how="left").with_columns(
    sw_mvt_first=pl.col("to") - pl.col("t_first_sw"),
    sw_mvt_first_move=pl.col("to") - pl.col("t_first_move_sw"),
    sw_extra_s=pl.col("t_first") - pl.col("t_first_sw"),
).with_columns(sw_recovered=(pl.col("sw_extra_s").fill_null(0) > 0).cast(pl.Int8))

missing = [c for c in NAMES + NEW if c not in d.columns]
if missing:
    sys.exit(f"нет колонок: {missing} — сначала переизвлечение с признаками визита и геометрии")
print(f"строк {d.height}; продлён отрезок у {int(d['sw_recovered'].sum())}, "
      f"визит известен у {int(d[NAMES[0]].is_not_null().sum())}, геометрия E66 у {int(d[NEW[0]].is_not_null().sum())}", flush=True)

for p in (CTRL_DIR, CAND_DIR):
    os.makedirs(p, exist_ok=True)
for a, b in [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]:
    tr, te = d.filter(~pl.col("month").is_in([a, b])), d.filter(pl.col("month").is_in([a, b]))
    y = te[E36.Y].to_numpy()
    line = []
    for name, out, feats in (("ctrl", CTRL_DIR, E36.FEATS + NAMES), ("cand", CAND_DIR, E36.FEATS + NAMES + SW + NEW)):
        pred = E36.correct(E36.fit_stage(tr, feats), te)
        np.savez(f"{out}/fold_{a}_{b}.npz", mvt_id=te["MVT_ID_mvt"].to_numpy(), pred=pred, y=y)
        line.append(f"{name} RMSE {np.sqrt(np.mean((pred - y) ** 2)):.2f}")
    print(f"fold {a}+{b}: " + " | ".join(line), flush=True)

dec, tot, small, trim, lower, n, strict = compare(CTRL_DIR, CAND_DIR)
ok = n == 6 and -dec >= 0.3 and lower >= 4 and -small >= 0.3 and -trim >= 0.3
print(f"DECISION E67: {'ACCEPT' if ok else 'REJECT'} (folds {n}, lower in {lower}; decided rows {dec:+.2f} s, "
      f"strict {strict:+.2f} s, all rows {tot:+.2f} s, y<=7200 {small:+.2f} s, trimmed {trim:+.2f} s)")
