# E187 Task 1 implementation report

Реализованы только `src/e187_weight_objective.py` и `tests/test_e187_weight_objective.py`. Реальных OOF-подбора/оценок, обучения, изменений исходных массивов, журнала или коммитов не было. Прочитаны только схема FIN1b и формы/dtype одного NPZ из каждого типа входа для проверки совместимости. Действительный запуск разрешён только после коммита предрегистрации root.

## Протокол и артефакты

- Точный зарегистрированный grid из 1593 точек, включая нулевую; фильтрация осей до itertools сохраняет первоначальный порядок. Сумма ≤1, шаг 0.05, шесть фиксированных box. При равенстве выбирается первая точка.
- Две цели: all SSE и decided AND y≤7200 SSE. Каждый held-out fold выбирает веса явной суммой пяти других folds; вычитание held-out SSE из общего SSE не используется.
- Float64, куски 8192 строк ×64 весов; ожидаемая смесь аффинна по reg, но LOBT и nonnegative clipping выполняются для каждой пары строка/вес. Это точная кусочная функция, не квадратичное приближение. Перестановка float64 сумм может давать округление; обязательная паритетная проверка saved E172 допускает maxabs 1e-6 секунды для pred/reg, проверяет восстановленные веса и явно помечает неоднозначность rank-deficient матрицы.
- `DAY` и актуальный `LOBT_WINDOW` читаются из literal AST `train.py` без импорта LightGBM. Сохраняется исходная разница между stored pred и восстановленной control mixture. Сохранённые `values[:,0/-1]` заново построены из нового reg в float64.
- Before metrics проверяется паритет всех шести saved-reference folds. ID/target/month/full-training-coverage проверяются до вычисления SSE. Legacy float IDs принимаются только как точные целые, безопасные для преобразования. Исходные nonregular NaN-кандидаты допустимы по математике E172; бесконечные значения и нечисловые/неполные прогнозы отвергаются.
- Результаты: отдельные `all_selected`, `body_selected`, `fixed_production_pre_tail`, scores NPZ, полный JSON manifest, краткий Markdown. Fresh output directory, exclusive lock, запись без перезаписи; manifest — последняя атомарная метка готовности. Partial directory остаётся без completed manifest и не используется повторно.
- SHA256 всех источников/исполняемого кода, константы, определения masks, веса каждого fold, исходные SSE и LB-equivalent deltas, gate flags, protected/nonprotected tails, месяцы/аэропорты/NM slices, вторичные fixed-weight comparisons. После расчёта исходные hashes проверяются повторно; outputs также имеют SHA256.
- Exit 0: валидный прошедший gate; 1: валидный отклонённый эксперимент; 2: invalid execution. Fixed production comparison явно pre-tail OOF, не v1139: нет CatBoost/seeds/final stages.

## Проверки

Первый запуск synthetic tests упал на отсутствующем API нового модуля до реализации. После реализации и исправления синтетических expected assertions:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 POLARS_MAX_THREADS=2 .venv/bin/python -m unittest discover -s tests -p test_e187_weight_objective.py -v
```

**10 tests PASS**, последний отдельный прогон 0.768 s. Проверяют scalar clipped mixture/delta/NaN, exact grid order/constraints/ties, tail-dependent selection, held-out score independence, chunked SSE parity, member/reference row alignment, sixfold completeness, duplicate/mismatched/NaN/Inf/month/coverage rejection, saved-reference failure/no-clobber, полный synthetic run/rejected gate и synchronized outputs.

Полный текущий synthetic suite:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 POLARS_MAX_THREADS=2 .venv/bin/python -m unittest discover -s tests -v
```

**56 tests PASS**, 1.641 s. `git diff --check` — без ошибок. Новых зависимостей/изменений `.venv` нет.

### Исправление P2 из независимого ревью

Оба воспроизведения подтвердились до исправления subprocess-тестом: string `pred` и `b'PK\x03\x04broken'` возвращали exit 1 с traceback. Причина — `np.isfinite` до dtype-проверки и отсутствие нормализации `zipfile.BadZipFile`.

Добавлена явная проверка real numeric dtype перед numpy ufunc/comparison для IDs, pred/reg/y/proba/values, member reg2/optional targets/months и feature target/month/LOBT/schedule. NaN разрешён только в предусмотренных кандидатах/nullable delays. Reader переводит BadZipFile в InvalidInput, требует именно NPZ и отвергает duplicate array names. KeyboardInterrupt/SystemExit и случайные programming exceptions не перехватываются широким except.

После исправления оба subprocess воспроизведения возвращают **exit 2**, строку `INVALID`, без traceback и completed manifest. Тестами дополнительно проверены все relevant numeric arrays, NPY под именем NPZ и duplicate archive names; новые проверки сначала продемонстрировали исходный сбой.

Актуальная проверка после исправления: **13/13 собственных synthetic tests PASS**, 1.733 s; **61/61 полного suite PASS**, 2.954 s; `git diff --check` чистый. Реальные OOF по-прежнему не оценивались; коммит/журнал не изменены.

## Ограничения для root/reviewer

1. Leave-one-fold-out выбор весов исключает прямое использование held-out labels, но **не является fully nested stacking**: модели, построившие OOF остальных folds, могли обучаться на labels текущего held-out fold. Это диагностический зарегистрированный E187; положительный результат нельзя выдавать за независимую внешнюю оценку или гарантированный прирост доски.
2. До настоящего запуска не проверены saved-reference parity и runtime на полном OOF. Несовпадение блокирует completed manifest; протокол/веса нельзя менять после просмотра результатов.
3. Память ограничена row×grid чанками, но точный перебор 1593 весов на полном OOF всё равно выполняет несколько миллиардов операций. После preregistration run root отслеживает fold progress; max threads=2.
4. Первоначальная float32 `values` может быть stale — выход исправляется, исходник остаётся нетронутым. Regular/day values могут измениться даже при нулевом выбранном векторе, сохраняя точный pred.

Команда после preregistration commit:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 POLARS_MAX_THREADS=2 .venv/bin/python src/e187_weight_objective.py --repo-root /Users/denyskolomiiets/prc-2026 --output /Users/denyskolomiiets/prc-2026/data/e187_<fresh_run_id>
```
