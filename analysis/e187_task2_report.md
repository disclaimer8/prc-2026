# Task 2 — защита E130 ranking

Готово в отдельном checkout; изменены только `src/e130_rank.py`, `src/rank_artifact.py`, `tests/test_rank_artifact.py` и этот отчёт. Основной checkout, реальные прогнозы и очереди не изменялись; LightGBM fits не запускались.

`E130_OUT` задаёт новый final. FIN1b требует явного пути, отличного от legacy final (включая разрешение symlink). Проверка final и эксклюзивный `<final>.lock` выполняются до подготовки признаков. **Существующий final всегда отклоняется; `E130_OVERWRITE=1` тоже отклоняется до подготовки признаков** с требованием выбрать fresh E130_OUT. Final публикуется только атомарным `link` без замены.

Независимый reviewer обнаружил гонку старого overwrite: legacy writer, не соблюдающий lock, мог заменить существующий final во время fit, после чего наш `os.replace` уничтожил бы его новый результат. По решению координатора in-place overwrite полностью удалён до предрегистрации; условная проверка stat/hash не заменяет атомарный compare-and-swap. API-параметр `overwrite` сохранён для явного отказа при True.

После каждого seed сохраняется только `<stem>.checkpoint.npz` со статусом checkpoint и `complete=False`. Final публикуется из полностью записанного временного файла только при точном числе seeds. Он сохраняет `mvt_id/reg/per_seed` и добавляет complete/status, expected/completed seeds, 344841 expected rows, путь/SHA256 признаков, mode/rounds и JSON provenance (хеши источников, параметры, колонки, категории, seed schedule, явный список настройки). Проверяются shape, конечность и уникальные целочисленные IDs; float IDs за пределами ±2^53 отклоняются, как strict validator. Числа seeds/rounds ≤0 запрещены. При исключении прежний final не меняется; stale lock после SIGKILL автоматически не снимается.

Перед публикацией SHA256 признаков пересчитывается; изменение источника во время fit запрещает final, checkpoint остаётся отдельным. Формулы fit, label, encoding, anchor, clip, параметры/seed schedule, predict и среднее seeds сохранены. IDs берутся после encoding, как в исходном скрипте; дополнительная ранняя проверка ловит неверное покрытие до подготовки encodings.

## Проверки

Команда RED/GREEN: `.venv/bin/python -m unittest discover -s tests -p test_rank_artifact.py -v`.

- RED: 11 failures — writer отсутствовал; после helper 9 pass/2 failures — скрипт читал parquet до FIN1b/existing-final guard.
- GREEN: 11/11 после интеграции.
- Дополнительный RED: symlink FIN1b обходил guard; исправление → 14/14.
- Дополнительный RED: `overwrite=True` при изначально отсутствующем final перезаписывал появившийся файл; исправление → 15/15.
- После ревью RED: `test_unsafe_floating_ids_reject_but_small_integral_floats_pass` принимал ID 2^53+2; исправление → 16/16 Task 2.
- Финальное ревью RED: `test_preexisting_final_rejects_even_with_overwrite_permission`, `test_script_existing_final_with_overwrite_flag_rejects_before_preparation`, `test_overwrite_permission_rejects_before_reading_features` обнаружили сохранённый небезопасный overwrite; `test_changed_feature_source_cannot_publish` обнаружил публикацию после изменения feature source. После удаления overwrite и повторного hash — **18/18 Task 2 pass**.
- Итоговая полная discovery `.venv/bin/python -m unittest discover -s tests -v` после завершения остальных задач — **58/58 pass**, 4,01 s.
- `.venv/bin/python -m py_compile src/e130_rank.py src/rank_artifact.py tests/test_rank_artifact.py` — pass.
- `git diff --check -- src/e130_rank.py src/rank_artifact.py tests/test_rank_artifact.py` — pass.

Ранняя полная синтетическая discovery во время работы остальных агентов — 50 tests, 29 failures/1 error; Task 2 был зелёным. Ниже сохранены ранние ошибки; **в итоговом запуске все исправлены владельцами**, чужие файлы не исправлялись:

- `test_e187_weight_objective.test_exact_protection_mask_and_metrics_gate`: protected_tail SSE 0 вместо −9.
- `test_e187_weight_objective.test_reference_parity_fails_before_completed_publication_and_no_clobber`: NumPy bool не сериализуется JSON.
- Все 28 тогда существовавших `test_strict_oof_gate` отклонялись в setUp, поскольку standalone module ещё отсутствовал: `test_both_saved_targets_wrong_against_features_reject`, `test_candidate_id_set_must_be_exact`, `test_cli_has_distinct_statuses_and_machine_readable_evidence`, `test_duplicate_id_across_folds_rejects`, `test_duplicate_id_within_fold_rejects`, `test_extra_fold_rejects`, `test_feature_training_ids_are_unique`, `test_four_of_six_requirement_blocks_three_improving_folds`, `test_fractional_and_unsafe_float_ids_reject`, `test_identical_inputs_pass_only_audit_mode`, `test_incomplete_training_coverage_rejects`, `test_incorrect_candidate_target_rejects`, `test_invalid_optional_array_shapes_and_nonfinite_reject`, `test_legacy_float_ids_are_accepted_when_exactly_integral`, `test_malformed_npz_rejects`, `test_missing_authoritative_feature_target_rejects`, `test_missing_sixth_fold_rejects`, `test_nonfinite_target_and_prediction_reject`, `test_object_array_rejects_without_pickle`, `test_rank_features_do_not_count_toward_training_coverage`, `test_require_body_can_reject_a_tail_driven_gain`, `test_shapes_and_missing_fields_reject`, `test_shuffled_candidate_rows_are_aligned_by_id`, `test_stale_values_are_reported_not_rejected_and_nan_plans_allowed`, `test_valid_improvement_and_hand_calculated_legacy_metrics`, `test_valid_worse_candidate_is_rejected`, `test_wrong_feature_month_rejects_even_without_saved_month`, `test_wrong_stored_month_rejects`.

## Будущая команда E182 (не запускалась)

После разрешённого gate, при прежнем зарегистрированном DROP_PREFIX и заданном `E182_RAW_ROUNDS` (среднее вложенных E182 rounds), из основного репозитория:

```sh
FEATURES=data/features_fin1b.parquet DROP_PREFIX=day_,tq_,wxr_,month,te_,ter_,STAND_mvt,ap_stand,rwy_config PROXY_SCALE=1 E130_MODE=raw E130_OUT=submissions/e130_rawf_rank.npz OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 POLARS_MAX_THREADS=2 .venv/bin/python src/e130_rank.py 5 "${E182_RAW_ROUNDS:?set the preregistered E182 raw mean rounds}"
```

Файл должен быть новым. Другие MODE требуют своих distinct имён/зарегистрированных rounds. Команда не выбирает параметры E182 и не меняет его протокол.
