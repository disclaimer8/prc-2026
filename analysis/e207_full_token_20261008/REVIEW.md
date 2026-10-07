# E207: независимое ревью перед пререгистрацией

Зафиксировано **07.10.2026 23:30:20 UTC**, до реального extraction/prep, GPU smoke и target-fit.

**Вердикт независимого reviewer: FINAL READY; 0 Critical, 0 Important.** Root подтвердил завершение ревью и независимый PASS всех11 targeted tests. Хеши автора и reviewer совпали; при записи этой квитанции все5 SHA-256 повторно сверены с локальными файлами. Это разрешение зафиксировать пререгистрацию и перейти к предусмотренным протоколом проверкам исполнения, не результат эксперимента.

База review: `4812d85b4cc2a65d02da9165a81bbac97fb1f109`. Текущий research HEAD: `f91b858c56f57d386efee21b226bfe5b3f3683ed`. Эти commit не являются будущим prereg commit: root должен закрепить код, протокол, тесты, квитанцию и журнал №377 одним отдельным commit до запуска.

Область review — новый sampler, controller/queue, targeted tests и соответствие [PROTOCOL.md](PROTOCOL.md), включая сохранение старых raw-индексов, source/control contracts, train-only normalization, inherited mixture coefficient, фиксированный gate, порядок extraction→prep→admission→synthetic smoke→fit, шестичасовой предел и cleanup собственных процессов. Frozen upstream не изменяется. [readiness.json](readiness.json) — авторский снимок23:25:50UTC до окончания независимого review; его статус `ready_for_independent_review_not_launched` сохранён как история.

| Frozen файл | SHA-256 |
|---|---|
| `scripts/e207_full_token_sampler.py` | `2c85fed68950bda537f130bfd7c4ed87aa4288d109ca5b85a2579c185e78d1d2` |
| `scripts/e207_full_token_queue_20261008.py` | `0de8fc67b1eb7ce81436eac5633a0a8488bffec6d0c4d4290193231814476ef6` |
| `tests/test_e207_full_token_sampler.py` | `5385c3483d61ef101d711219ef5b078a6ff6fb9b2622a9bf51a0ab4917c5cf06` |
| `analysis/e207_full_token_20261008/PROTOCOL.md` | `a71fb27d11b41df29473939d6dafa7dd4e62fea3eb6f0ec7c0f490661c5bbb67` |
| `analysis/e207_full_token_20261008/metadata.json` | `535f8ba342f944e08466a1d5e0faaa7ef6ba9eed2f8466455cb511780897b423` |

Независимо пройденная команда: `.venv/bin/python -m unittest discover -s tests -p test_e207_full_token_sampler.py` — **11 PASS, 0 FAIL**. Авторский readiness также фиксирует compile PASS и соседний набор `test_night_gpu_20261007.py`:6 PASS/1 SKIP (qualified PLE package находится на Home). Этот skip не заменён заявлением об успешном GPU smoke: **реальный full-shape synthetic GPU smoke ещё не выполнен** и обязателен перед первым target-fit. Повторных тестов при оформлении этой квитанции не запускали.

Единственный новый arm F сравнивается с saved ordered O на6 парах,3сида×8эпох: **18 target-fit**, без новых control-fit. Все старые raw-индексы сохраняются; свободные позиции заполняются детерминированно из оставшихся raw-индексов. Прежние правила PAD/visit masks сохраняются, actual token-mask меняется. Точный neural coefficient —23/120 на E185 subset, .25 на остальных строках. Gate: decided ΔSSE<0 минимум в4/6пар, pooled body/decided ΔSSE<0, суммы within-fold decided symmetric trim3/trim20 ΔSSE<0. Prior body ΔRMSE=0с, экспертный диапазон[−1,+1]с, не доверительный интервал.

Ограничения: повторные timestamps и плотность приёма способны ухудшить перенос; metadata не измеряет пользу. Это selection-informed LGB5/preADSB component comparison с историческим upstream, не full-current/nested CV. Даже PASS требует отдельной пререгистрации current-E203 preE81 chain; **ranking и upload запрещены**, `accepted_for_submission=false`. На этой отметке **0 real extraction/prep, 0 target-fit**, время и качество нового опыта не измерены. CPU171/175 и HomeGPU запускаются только после root prereg commit в рамках протокола с единым hard deadline6ч, без retry/resume/fallback.
