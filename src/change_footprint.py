"""След изменения: сколько строк оно трогает и сколько оставляет нетронутыми (18.09).

    .venv/bin/python src/change_footprint.py <контрольный parquet> <кандидатский parquet> [колонки...]

Зачем. Два правила, выведенных 17–18.09, говорят об одном: цена изменения лежит не в новых строках, а в реакции
модели на них.
  * **Поглощение** (разд. 21f брифа): модель сама выражает добавку имеющимися признаками — изменение стоит ноль.
  * **Переучивание** (E61): часть строк меняет распределение признака, модель перестраивается, и нетронутое
    большинство платит за это. У E61 соотношение было 111 тыс. изменённых против 1,97 млн нетронутых, и
    нетронутые дали +0,261 с против −0,063 с у изменённых.

Второе считается **до** прогона: достаточно сравнить два входа и посмотреть, сколько строк меняется, сколько
остаётся, и какую долю SSE держит каждая группа. Печатается три числа по каждой колонке (появилось / исчезло /
изменилось), потому что один фильтр `!=` в polars теряет появления: сравнение с null даёт null.

Если задан `BASE_DIR` с сохранёнными прогнозами фолдов, рядом печатается доля SSE, которую держат тронутые и
нетронутые строки — то есть чем именно рискует переучивание.
"""
from __future__ import annotations

import glob
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

KEY = os.environ.get("KEY", "MVT_ID_mvt")
BASE_DIR = os.environ.get("BASE_DIR", "data/cv_preds_e46")
LB = 6 * 2 * 252.43 * 344841


def main() -> None:
    ctrl_path, cand_path = sys.argv[1], sys.argv[2]
    # путь может быть шаблоном: тогда берутся все совпавшие сутки, но с обеих сторон — один и тот же набор,
    # иначе контроль и кандидат считаются на разных днях
    def load(path, other):
        if '*' not in path:
            return pl.read_parquet(path)
        names = {os.path.basename(f) for f in glob.glob(path)} & {os.path.basename(f) for f in glob.glob(other)}
        d = os.path.dirname(path)
        return pl.concat([pl.read_parquet(os.path.join(d, n)) for n in sorted(names)])
    a = load(ctrl_path, cand_path)
    b = load(cand_path, ctrl_path)
    cols = sys.argv[3:] or [c for c in a.columns if c != KEY and c in b.columns]
    j = a.select([KEY, *cols]).join(b.select([KEY, *cols]), on=KEY, suffix="_new", how="full", coalesce=True)
    print(f"строк: контроль {a.height}, кандидат {b.height}, объединение {j.height}\n")
    touched = pl.lit(False)
    for c in cols:
        x, y = pl.col(c), pl.col(f"{c}_new")
        appeared = j.filter(x.is_null() & y.is_not_null()).height
        lost = j.filter(x.is_not_null() & y.is_null()).height
        changed = j.filter(x.is_not_null() & y.is_not_null() & (x != y)).height
        print(f"  {c:20s} появилось {appeared:7d}   исчезло {lost:7d}   изменилось {changed:7d}")
        touched = touched | x.ne_missing(y)
    t = j.with_columns(touched=touched)
    n_t = int(t["touched"].sum())
    print(f"\nтронуто строк {n_t} ({n_t / j.height * 100:.2f} %), нетронуто {j.height - n_t}")

    files = sorted(glob.glob(f"{BASE_DIR}/fold_*.npz"))
    if not files:
        print(f"\n{BASE_DIR}: прогнозов фолдов нет, доля SSE не считается")
        return
    parts = []
    for p in files:
        z = np.load(p)
        parts.append(pl.DataFrame({KEY: z["mvt_id"], "e2": (z["pred"] - z["y"]) ** 2}))
    e = pl.concat(parts).join(t.select(KEY, "touched"), on=KEY, how="inner")
    sse = float(e["e2"].sum())
    on = float(e.filter(pl.col("touched"))["e2"].sum())
    print(f"\nSSE (по {BASE_DIR}, {e.height} строк): всего {sse / LB:.2f} с")
    print(f"   тронутые строки:   {int(e['touched'].sum()):8d}  держат {on / sse * 100:5.2f} % SSE = {on / LB:6.2f} с")
    print(f"   нетронутые строки: {int((~e['touched']).sum()):8d}  держат {(1 - on / sse) * 100:5.2f} % SSE = {(sse - on) / LB:6.2f} с")
    print("\nЧитать так: выигрыш может прийти только с тронутых строк, а переучивание облагает налогом все.")


if __name__ == "__main__":
    main()
