"""E87: обучать логистику E14 на квадрате ошибки её собственного выхода, а не на классификации (21.09).

    .venv/bin/python src/e87_e14_sse.py

E14 применяется как смесь двух кандидатов: ŷ = r + p(x)·(s − r), где r — обычное руление, s = D — кандидат
расписания. Но p(x) обучается логистической регрессией на бинарной метке «это строка расписания», то есть на
правдоподобии класса. Платим же мы за квадрат ошибки смеси. Эти две задачи расходятся тем сильнее, чем
дальше друг от друга кандидаты: при |s − r| = 3 ч та же десятая доля вероятности стоит в десять тысяч раз
дороже, чем при |s − r| = 100 с, а логистическая потеря этого не видит.

E87 оптимизирует ровно то, за что платят:

    L(w) = Σ_i [ y_i − r_i − σ(x_i·w)·(s_i − r_i) ]²

Признаки, популяция и правило применения — те же, что у E14; старт оптимизации берётся из логистического
решения, так что при w = w_логит потеря не хуже исходной по построению.

Отличие от закрытого: E23 выбирал число раундов, E58 калибровал частоту класса, E86 взвешивает
классификационную потерю. Здесь меняется **сама обучаемая задача**, и меняется у той модели, которая
действительно определяет ответ на этих строках: основной классификатор на них перезаписывается E14.

ЗАПИСАНО ДО ПРОГОНА. Проверка: вероятности и веса подбираются на десяти месяцах, применяются к двум
отложенным, шесть фолдов по парам месяцев — тот же протокол, которым принимался E14. Ворота: SSE на строках
применения ниже на **≥ 0,30 с** в единицах ворот и ниже в **≥ 4 из 6** фолдов. Прогноз: **−0,05 … −0,60 с,
центр −0,20**. Центр скромен: популяция мала (около 1 300 строк), логистика на ней уже откалибрована в
среднем (q 46,0 % против истины 46,1 %), и выигрыш может дать только перераспределение внутри неё.
"""
import os
import sys

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

LB = 6 * 2 * 243.8347 * 344841
FOLDS = [(1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]

src = open("src/e14_flight_history.py").read().split("\nresults = []")[0]
g = {"__name__": "e14"}
exec(src, g)
f, history, design, Y = g["f"], g["history"], g["design"], g["Y"]
WINDOW = (pl.col("d_mvt_sched") > 0) & (pl.col("d_mvt_sched") < 14400)


def regular_candidate() -> dict:
    """r — кандидат «обычное руление» из сохранённых OOF принятой цепочки (колонка 0 массива values)."""
    out = {}
    for a, b in FOLDS:
        d = np.load(f"data/cv_preds_e77_stage/fold_{a}_{b}.npz")
        out.update(dict(zip(d["mvt_id"].tolist(), d["values"][:, 0].tolist())))
    return out


def sse_fit(X: np.ndarray, y: np.ndarray, r: np.ndarray, s: np.ndarray, w0: np.ndarray) -> np.ndarray:
    def loss(w):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        return float(np.mean((y - r - p * (s - r)) ** 2))
    res = minimize(loss, w0, method="L-BFGS-B", options={"maxiter": 500})
    return res.x if res.fun <= loss(w0) else w0


def main() -> None:
    reg = regular_candidate()
    parts = [history(f.filter(pl.col("month") != m), f.filter(pl.col("month") == m)) for m in f["month"].unique().to_list()]
    pop = pl.concat(parts).filter(WINDOW).filter(pl.col("n") >= 3)
    print(f"популяция применения: {pop.height} строк")

    tot_old = tot_new = 0.0
    lower = 0
    print(f"\n{'фолд':8} {'строк':>6} {'SSE логит':>11} {'SSE по SSE':>12} {'разница, с':>11}")
    for a, b in FOLDS:
        tr = pop.filter(~pl.col("month").is_in([a, b]))
        te = pop.filter(pl.col("month").is_in([a, b]))
        if te.height == 0:
            continue
        Xtr, Xte = design(tr), design(te)
        ytr, yte = tr[Y].to_numpy().astype(float), te[Y].to_numpy().astype(float)
        str_, ste = tr["d_mvt_sched"].to_numpy().astype(float), te["d_mvt_sched"].to_numpy().astype(float)
        rtr = np.array([reg.get(m, np.nan) for m in tr["MVT_ID_mvt"].to_list()], dtype=float)
        rte = np.array([reg.get(m, np.nan) for m in te["MVT_ID_mvt"].to_list()], dtype=float)
        ok_tr, ok_te = np.isfinite(rtr), np.isfinite(rte)
        Xtr, ytr, str_, rtr = Xtr[ok_tr], ytr[ok_tr], str_[ok_tr], rtr[ok_tr]
        Xte, yte, ste, rte = Xte[ok_te], yte[ok_te], ste[ok_te], rte[ok_te]
        if len(yte) == 0:
            continue

        lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, tr["sched"].to_numpy()[ok_tr])
        Xtr1 = np.column_stack([Xtr, np.ones(len(Xtr))])
        Xte1 = np.column_stack([Xte, np.ones(len(Xte))])
        w0 = np.append(lr.coef_.ravel(), lr.intercept_[0])
        w = sse_fit(Xtr1, ytr, rtr, str_, w0)

        p_old = lr.predict_proba(Xte)[:, 1]
        p_new = 1.0 / (1.0 + np.exp(-np.clip(Xte1 @ w, -30, 30)))
        e_old = float(np.sum((yte - (rte + p_old * (ste - rte))) ** 2))
        e_new = float(np.sum((yte - (rte + p_new * (ste - rte))) ** 2))
        tot_old += e_old; tot_new += e_new; lower += e_new < e_old
        print(f"{a}+{b:<5} {len(yte):6} {e_old / LB:11.2f} {e_new / LB:12.2f} {(e_new - e_old) / LB:+11.2f}")

    delta = (tot_new - tot_old) / LB
    ok = -delta >= 0.30 and lower >= 4
    print(f"\nИТОГО: {delta:+.2f} с, ниже в {lower} из 6")
    print(f"DECISION E87: {'ACCEPT' if ok else 'REJECT'}")


if __name__ == "__main__":
    main()
