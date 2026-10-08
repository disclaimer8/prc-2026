# E207 v2: review технического исправления NPZ

Срез **08.10.2026 00:10:59 UTC**, до реального запуска v2. Автор подтвердил frozen-состояние; независимое scoped-review сообщает **0 Critical / 0 Important**, targeted suite **12 PASS / 0 FAIL**. Root отдельно повторил эти12 тестов с PASS. При фиксации квитанции все6 SHA из [readiness.json](readiness.json) повторно совпали с локальными файлами. Финальный prereg commit и запуск принадлежат root; здесь запусков нет.

Родитель — №377/`741045e9fe10c17f68a7ce6bed4216e6160ee7b1`. V1 технически остановлен: owned175/171 cleanup00:07:42.600974/00:07:43.351569UTC, `failed_no_retry`00:07:51.862671UTC; оба unit inactive/MainPID0, **0 completed-day receipts,0 prep,0 target-fit**. [История и доказательство остановки](../e207_full_token_20261008/STATUS.md), [исходные events](../e207_full_token_20261008/execution/events.jsonl). Это не научный FAIL F-only.

Единственная функциональная поправка: compressed NPZ один раз загружается в dict, затем передаётся в неизменный row-wise validator; каждый из13 members распаковывается один раз. [Нормализованный scoped diff](scoped_diff.txt) не содержит изменения sampler/признаков/model/gates; controller отличается v2 namespaces, source names и идентификатором регистрации. Старые source/protocol/partials сохранены, v2 использует свежие каталоги/unit names.

[Reproduction](reproduction.json) на настоящем compressed NPZ с **синтетическим содержимым** измеряет исходные24/114/969 member reads при1/7/64 строках. Readiness подтверждает вызов настоящего `extract_day`:13 reads, каждый key ровно1, при1 и7 строках, RED до исправления/GREEN после. Дополнительно независимый reviewer передал root проверку64 строк: **969→13 reads**, arrays/dtype/shape/stats exact. Это проверка числа чтений и эквивалентности; не измеренный speedup всего реального extraction и не результат RMSE.

| Frozen файл | SHA-256 |
|---|---|
| [e207_full_token_sampler_v2.py](../../scripts/e207_full_token_sampler_v2.py) | `52867eb1772ed4bdcfc9088e097e5d2e12d72736796e61ab67dc7ecb7107a452` |
| [e207_full_token_queue_v2_20261008.py](../../scripts/e207_full_token_queue_v2_20261008.py) | `5ed43d38c6d669b4766bd25f3a9d6aee196aab9b9133e8cf835e77417eb90a13` |
| [test_e207_full_token_sampler_v2.py](../../tests/test_e207_full_token_sampler_v2.py) | `f8f0e94e65d7e28d12094ead6f5da80aecdd5f59b670f871ff6d502ae5789132` |
| [PROTOCOL.md](PROTOCOL.md) | `7601e9e50feb905a87ed32d03c6b71248a56736a1572dedbde94245da092a6d6` |
| [reproduction.json](reproduction.json) | `6b3c8018ca0ef1819abb27cff64471cb4edb4012aa9e38dda688cbf7ad034ff0` |
| [scoped_diff.txt](scoped_diff.txt) | `e726e091007b3909a87b23b82a8c5d3e79feb561a696c83e7078cebdb4b89b0e` |

Команда targeted suite: `.venv/bin/python -m unittest discover -s tests -p test_e207_full_token_sampler_v2.py`. Compile PASS по авторскому readiness. При оформлении квитанции тесты заново не запускались. Реальные v2 extraction, normalization/prep, full-shape GPU smoke и target-fit ещё не выполнялись; synthetic GPU smoke остаётся обязательным перед первым fit.

[Полный протокол](PROTOCOL.md): `E207-full-K64-retain-original-raw-index-fill-sixpair-3seed8epoch-v2`. Научный рецепт №377 неизменен: F retaining old indices против saved ordered,6пар×3сида×8эпох=18fit; inherited coefficient23/120 на E185 subset/.25 иначе; train-only normalization RULE; ≥4/6 decided улучшений, pooled body/decided и within-fold decided trim3/trim20 ΔSSE<0. Prior0с/экспертный[−1,+1]с прежний. Selection-informed LGB5/preADSB comparison требует отдельного current-E203-chain протокола до ranking.

Новые v2 namespaces, CPU171/175→HomeGPU, прежние resource guards и единый hard6ч; без retry/resume/fallback. **Пререгистрация №378 предшествует реальному запуску v2; ranking/upload не разрешены.** Ограничения reception-density transfer и отсутствия полного nested/current-tail OOF не изменились.
