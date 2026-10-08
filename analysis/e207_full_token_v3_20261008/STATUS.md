# E207 v3: оперативный статус

Проверено08.10.2026 11:10UTC. Это snapshot; события — `execution/events.jsonl`.

E207 v3 зарегистрирован №379, commit `f43405e7c6836230862837b66fad081ae4d69777`.
Фактический старт controller55670 —08.10 10:06:07.756092UTC. Все365 дней и6 fold
таблиц готовы. Materialized spec `56e1417bb71aa917847a7536ffbf8f0a9110c768a66bbaa6bf32e851fbffc4ca`.
Synthetic full-shape GPU smoke PASS (flat/ordered, peak13.02/12.75GB);
[квитанция](analysis/e207_full_token_v3_20261008/smoke_snapshot.json).
Fold1 завершён к11:01:50UTC: **3/18 fit завершены**. На11:10UTC идёт второй фолд
(seed1 epoch4/8); это обучение, не оценка качества. Folds1/2/3 допущены,
fold4/5 ещё передаются. GPU-unit PID870055, Invocation`4ccf79f86a144e4d9aaedd1ef0f169bd`,
active/running. Не менять frozen source/protocol и не дублировать очередь.
После всех6пар/18fit нужен отдельно зарегистрированный current-chain CV; нового RMSE нет.
Best230.9387;15/237 — полная доска08.10 10:31:23UTC, authenticated квота08.10 не проверена.
E208 №380 реально обрабатывает фиксированный16-дневный label-free пилот наHetzner.
Пакет saved-components v12075 №381 зарегистрирован, **не запущен и отложен** ради
единого финального конвейера. E201 отменён; ARR-пилот остановлен до extraction; LLMoff.

## Подтверждённые этапы

| Этап | Доказательство |
|---|---|
| Старт controller |10:06:07.756092UTC, PID55670, launch.json |
| Raw171 |10:07:51→10:10:55UTC, InvocationID a39aa0fb8b4446b5a9f9b8050dc27afe, success0 |
| Raw175 |10:07:52→10:11:55UTC, InvocationID755ffa9d56fb40eda67c44a34273bdc1, success0 |
| Обмен raw и materialize |10:18:39UTC, все365days |
| Prep171/175 |Старт10:18:40/42UTC; логи содержат завершённые folds1/2/5 и3/4/6; на10:28 оба active/exited/MainPID0, peak10GiB |
| Передача/admission |Fold1 допущен10:50:06UTC, fold3 10:50:38, fold2 11:00:12; fold4/5 ещё передаются |
| GPU-unit |10:18:42UTC, PID870055, InvocationID4ccf79f86a144e4d9aaedd1ef0f169bd; обучение второго фолда, seed1 epoch4/8 на11:10UTC |
| GPU smoke |Flat/ordered full-shape PASS; receipt smoke_snapshot.json; peak13.02/12.75GB |
| Target-fit / RMSE |Fold1:3/18fit завершены к11:01:50UTC; второй учится. RMSE ещё не получен |

13 локальных regression tests PASS;6 bounded real-systemd smoke cases PASS.
Технические ошибки v1/v2 исправлены в отдельной preregistered v3, scientific recipe сохранён.
Очередь имеет общий hard deadline6h и отсутствие automatic retry. При новом техническом
сбое сохранить partials и исследовать причину. Frozen исходники не менять.
После успешного E207 нужен отдельно зарегистрированный CV-only current-chain child,
нулевые новые CPU fits; старый E206 draft нельзя запускать как E207.
