# E207 v3: проверка lifecycle и техническая пререгистрация

Срез **08.10.2026 03:43:23UTC**, до реальной v3 extraction/prep/fit. Автор подтвердил frozen READY; root проверил **13 targeted tests PASS** и фактический remote lifecycle smoke **6/6 PASS**. Все7 frozen SHA из [readiness.json](readiness.json) при записи этой квитанции повторно совпали. **Независимый reviewer: FINAL READY,0 Critical/0 Important; все7 SHA,13 tests и6 real lifecycle cases подтверждены.** Root получил финальный вердикт до prereg commit; реального v3 запуска здесь нет.

Родитель v2 — №378/`ea16e05330d3fe4fb5573dc17938002f14645876`. Его terminal02:28:53.186665UTC — `owned unit invocation changed` после успешного завершения shard171 и очистки transient unit системой. Сохранены183+145 дневных receipt,0 normalization/prep/fit. [V2 STATUS](../e207_full_token_v2_20261008/STATUS.md), [terminal events](../e207_full_token_v2_20261008/execution_snapshot.jsonl). Это техническая остановка, не научный FAIL.

Область поправки: `RemainAfterExit=yes`, явный `SubState`, распознавание retained `active/exited` с MainPID0 как terminal **при сохранённом exact InvocationID**; Result=success/ExecMainStatus0 обязательны. Пустая/чужая identity и nonzero exit остаются ошибкой. В prep-publication закрыта гонка: после successful terminal выполняется один повторный `file_ready`, если первая проверка ещё не увидела complete.json. Scientific worker отличается отv2 только namespaces; однократная загрузка NPZ сохранена. [Scoped diff](scoped_diff.txt), [полный протокол](PROTOCOL.md).

Фактический [lifecycle smoke](lifecycle_smoke.json) завершён **03:41:07.849750UTC**:171/175/Home×exit0/exit7=6cases. Проверены retained identity/terminal и отклонение exit7; все6 owned test units после cleanup inactive/MainPID0. Использованы только `/bin/sh`,sleep2,exit; CPU10%,RAM32MiB,swap0,RuntimeMax15с — без project data/models. Этот smoke проверяет systemd lifecycle, **не full-shape GPU smoke**, который ещё обязателен перед target-fit.

Targeted suite: `.venv/bin/python -m unittest discover -s tests -p test_e207_full_token_sampler_v3.py`,13 PASS/0 FAIL, RED до lifecycle-fix. Protocol дополнительно фиксирует10 сценариев nested launch/wait/publication, проверенных независимым reviewer. При оформлении квитанции тесты не повторялись. Готовые production units с retained active/exited могут остаться как status records: MainPID0 означает отсутствие вычисления; их автоматическое удаление не заявляется.

| Frozen файл | SHA-256 |
|---|---|
| [e207_full_token_sampler_v3.py](../../scripts/e207_full_token_sampler_v3.py) | `68ea0d07d61a9a5d998c89d0db2c40175e4a19bf9061a68e5436e16610d36cc9` |
| [e207_full_token_queue_v3_20261008.py](../../scripts/e207_full_token_queue_v3_20261008.py) | `894f77530de3e159a7c7513f470387cfc2afed11b36bafcd09f986b14ba9a8f4` |
| [test_e207_full_token_sampler_v3.py](../../tests/test_e207_full_token_sampler_v3.py) | `7f7cf0287f4fd929a8d382ad013734bc610e427ee13e004ef4501178881f9d1a` |
| [PROTOCOL.md](PROTOCOL.md) | `5243a217ef1002193fe4e291ee923fea645fc874544199209a68d9e1b6115baa` |
| [scoped_diff.txt](scoped_diff.txt) | `7d9a81ef7a5eb16a42e17a5677187dc3d10cc708a6b9ceaa6dbb1470724e5f1c` |
| [lifecycle_smoke.py](lifecycle_smoke.py) | `0d6d006859d9eb714c65e13a913b0c216fb1dc2c31a089a239ef9426a85af894` |
| [lifecycle_smoke.json](lifecycle_smoke.json) | `ae83e794c5000c89eab850a33f0b160818966b340bc612936bc6cd3d80d61a97` |

Регистрация: `E207-full-K64-retain-original-raw-index-fill-sixpair-3seed8epoch-v3`. Научный рецепт №377/378 неизменен: один F retaining-old arm против saved ordered,6пар×3сида×8эпох=18fit; coefficient23/120 на E185 subset/.25 иначе; прежние source channels/model/train-only normalization RULE/gates≥4/6 decided и pooled body/decided/within-fold decided trim3/20 ΔSSE<0; prior0с/экспертный[−1,+1]. Selection-informed preADSB, отдельный current-E203-chain до ranking.

Fresh v3 namespaces, новая extraction обоих shards без resume из v2 partials, CPU171/175→HomeGPU, прежние guards/hard6ч/no retry/resume/fallback. **0 реальных v3 extraction/prep/fit; ranking/upload не разрешены.** Root фиксирует №379/code/protocol/review до запуска. Старые source/protocol/outputs сохраняются.
