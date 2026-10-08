# Independent review: fresh final ordered trajectory member

Срез: **2026-10-08 11:50:22 UTC**. Reviewer: `fresh_external`.

**Вердикт: READY в проверенном scope; открытых Critical/Important/Minor нет.**
Условие финального freeze shared proxy снято: root подтвердил independent READY
этого отдельного адаптера, его SHA не изменился. В readiness установлен
`shared_proxy_final_freeze_pending=false`; reviewer повторно проверил все 15
source pins. SHA обновлённого readiness:
`31b44c980d8481e8b210076b780ae3ebe24d2cd06ce70e6c9f06ccfa0f75beb6`.
Source delta отсутствует; повтор обучения или дополнительных тестов не нужен.

## Scope и проверенная версия

Полностью прочитаны [worker](../../scripts/final_ordered_trajectory_20261008.py),
[tests](../../tests/test_final_ordered_trajectory_20261008.py) и
[PROTOCOL](PROTOCOL.md). Проверены используемые пути original E194 extraction,
E197 normalization/network/fit, original low-memory tabular preparation,
month-out target encoding, proxy и helper admission. Это review реализации
фиксированного original E203 member для нового final периода, не оценка нового
механизма и не свидетельство улучшения RMSE.

Base HEAD из [readiness](readiness.json):
`a725798a070896f2e1ec156e2d32b4f601c8a6ac`.

| Файл | Проверенный SHA-256 |
|---|---|
| Worker | `510f3d7c03455da75681e6b3cb5e1aa81dba787a7a5211ff0a957c7ddc3355f2` |
| Tests | `23a7b863de591e4f57ff2f7fb3deeba4b4a7f26e1a5daee9c80da4aa97825011` |
| Protocol | `3ab97f043d61356929310970b76ad1dd4e3efe770c0d89db184f91fe0e942456` |
| Shared proxy, final freeze подтверждён | `ba9a5e3aef9ef93b6d692806cba844c8fb0c1d075a6346245ac60c9ddf1bea37` |

Независимо пересчитаны и совпали **все 3 own pins и 12 helper pins** из readiness.
Последняя author delta — fail-closed проверка ID order после target encoding;
обе последовательности проверяются до повторного использования regime labels.

## Существенные проверки

- Используются original unfilled E194 sampler и original `S.fit(arm='ordered')`:
  три seed0/1/2, восемь эпох, unchanged network/optimizer/schedule. E207 sampler
  не подменяет этот member. Dynamic final population — 670,790 строк;
  old January/July predictions не являются входом.
- Query admission связывает Git registration/source closure с sealed feature
  manifest и completion. Проверяются actual feature/template hashes, ordered
  ID hashes, unique disjoint train/final IDs, 12 training months, split counts и
  null final Y/BLOCK. Normalization, PLE bins и подготовка модели используют
  ordinary 2025 population. 2026 DEP labels не допускаются.
- Сохраняются реально использованные TE/TER statistics, numeric quantile knots,
  missing indicators, actual categorical codes, proxy, residual scale и
  trajectory normalization. Код требует exact replay подготовленных tensors,
  original proxy parity, reload каждого seed checkpoint и exact equality final
  seed/mean predictions. Отдельный replay не обучает новые statistics/models.
- Raw 2025 reuse требует original E194 registration/source, canonical training
  hashes, input/output/day receipts, projected movements и surface SHA.
  Каждый 2026 day кодируется заново по новым MVT. Raw validation проверяет
  IDs, shapes/dtypes, masks, PAD, timestamp и future-point boundary;
  compressed NPZ members материализуются один раз на день.
- Fresh output namespace, artifact inventory/hash binding и failure marker
  предотвращают тихий overwrite/reuse partial outputs. Linux cgroup CPU/RAM,
  swap0, disk guards и 6h alarm присутствуют. GPU smoke связывает runtime и
  query receipt; durable service ownership, staging и освобождение GPU после
  E207 остаются обязанностью root, как указано в protocol.

## Независимая проверка и ограничения

Команда:

```text
.venv/bin/python -m unittest discover -s tests -p test_final_ordered_trajectory_20261008.py
```

**13/13 PASS**, повтор после guard delta: 8.691 s. Тесты включают original
preparer/serialized replay, null/rare categories, month-out/full TE/TER,
dynamic IDs, seed completeness, original duplicate-time sampler, raw future/PAD
rejection, ordinary-only normalization, raw lineage, inventory/failure guards
и PLE-bin persistence.

Reviewer не запускал actual preparation/raw encoding/GPU smoke/model fit/final
inference/replay, не читал реальные target arrays и не запускал full suite.
Тесты и static review не устанавливают фактическую память/ETA, real-data exact
parity, checkpoint equality на всех final rows или будущий RMSE/LB. Эти проверки
должны дать отдельные execution receipts после registration. Изменён только
этот review report; source/protocol/данные не редактировались.
