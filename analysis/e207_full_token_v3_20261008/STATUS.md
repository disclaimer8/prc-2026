# E207 v3: оперативный статус

Проверено08.10.2026 10:28UTC. Это snapshot, актуальные события — `execution/events.jsonl` и `controller.log`.

E207 v3 зарегистрирован №379, commit `f43405e7c6836230862837b66fad081ae4d69777`.
Фактический старт controller55670 —08.10 10:06:07.756092UTC; ожидание разрешения
до этого не является вычислением. Все365 дней извлечены, materialized spec
`56e1417bb71aa917847a7536ffbf8f0a9110c768a66bbaa6bf32e851fbffc4ca`.
На10:28UTC все6 fold tables подготовлены на171/175; оба prep-unit retained
active/exited/MainPID0. Идёт передача fold1/3 наHome; GPU-unit PID870055
active/running, лог `WAIT PREPARED 1`: **0 подтверждённых target-fit**, smoke ещё
не подтверждён. [Точный статус E207 v3](./execution_snapshot.jsonl).
Не менять frozen source/protocol и не дублировать очередь. Далее admission,
synthetic GPU smoke и6пар/18fit; ranking требует отдельного current-chain гейта.
Best230.9387 прежний;14/237 — снимок07.10 22:00UTC, live-квота08.10 не проверена.
E208 D-ATIS coverage и переносимый saved-component пакет v12075 готовятся;
их bulk-запуски ещё не зарегистрированы. E201 отменён, LLMoff.

## Подтверждённые этапы

| Этап | Доказательство |
|---|---|
| Старт controller |10:06:07.756092UTC, PID55670, launch.json |
| Raw171 |10:07:51→10:10:55UTC, InvocationID a39aa0fb8b4446b5a9f9b8050dc27afe, success0 |
| Raw175 |10:07:52→10:11:55UTC, InvocationID755ffa9d56fb40eda67c44a34273bdc1, success0 |
| Обмен raw и materialize |10:18:39UTC, все365days |
| Prep171/175 |Старт10:18:40/42UTC; логи содержат завершённые folds1/2/5 и3/4/6; на10:28 оба active/exited/MainPID0, peak10GiB |
| Передача |Fold3 10:19:57, fold1 10:19:59UTC; incoming наHome2.6/2.4GiB на10:28UTC, ещё неполные |
| GPU-unit |10:18:42UTC, PID870055, InvocationID4ccf79f86a144e4d9aaedd1ef0f169bd; пока WAIT PREPARED1 |
| Target-fit / RMSE |0 подтверждённых; результата пока нет |

13 локальных regression tests PASS;6 bounded real-systemd smoke cases PASS.
Технические ошибки v1/v2 исправлены в отдельной preregistered v3, scientific recipe сохранён.
Очередь имеет общий hard deadline6h и отсутствие automatic retry. При новом техническом
сбое сохранить partials и исследовать причину. Frozen исходники не менять.
После успешного E207 нужен отдельно зарегистрированный CV-only current-chain child,
нулевые новые CPU fits; старый E206 draft нельзя запускать как E207.
