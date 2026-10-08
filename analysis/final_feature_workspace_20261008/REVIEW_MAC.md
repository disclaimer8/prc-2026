# Independent review: Mac feature execution adapter

Срез **2026-10-08 12:03:42 UTC**. Reviewer `fresh_external`.

**FINAL READY: 0 открытых Critical/Important/Minor.** Оба найденных Important
исправлены и независимо перепроверены. Final adapter SHA:
`f1401dfb964c4a34fa1c98097b973f7d66ad4789fb1a03ffb622df828a9763a3`.
Protocol и generic feature worker не изменились.

Delta regression: canonical completion успешно проходит unchanged
`W.verify_completion`; семь intermediates удалены, четыре final outputs и
внешний old-file sentinel сохранены. Exited-worker failure вызывает TERM owned
group и пишет failed receipt. Прерывание seal вызывает TERM его отдельной
owned group и failed receipt. Дополнительно проверена bounded TERM→KILL ветка
с `wait(timeout=10)`. Все четыре проверки PASS в tempfile/mocks, без реальных
producers, signals чужим процессам или build. SIGTERM/SIGINT supervisor теперь
превращаются в exception и проходят тот же cleanup. Teardown вызывается и
после нормального завершения обоих owned children.

Review ограничен
[Mac adapter](../../scripts/final_feature_mac_20261008.py) и
[Mac protocol](MAC_PROTOCOL.md), совместимостью с frozen generic feature worker
`33479b6d8e8e9648b0c5c6410cba157f940ce6ca0389cd031f8164326f56b1be`.
Actual seal/build и реальные producers не запускались; source не менялся.

История findings, оба закрыты. Проверенная начальная версия: adapter
`ec1987ff7f51246346a816a8ba50984bfb4241b780298d43c34f50180b4c0ece`,
protocol `f674b6dceda7643b32521b1791edee6d7ac120d0483ab38e7a992080a60855df`.

1. **Закрыт: failure cleanup зависел от живого leader.** Если worker уже
   завершился, условие `child.poll() is None` пропускает `killpg`, хотя producer
   может оставаться в owned group. Synthetic supervisor с exited child,
   returncode1, дал `killpg.call_count=0` и failure receipt. Нужен teardown owned
   PGID независимо от leader status, включая bounded TERM/KILL и отсутствие
   группы. Также SIGTERM supervisor должен убирать seal child, запущенный до
   основного worker.
2. **Закрыт: completion receipt был несовместим с frozen verifier.** Дополнительный
   `execution_adapter` нарушает exact field contract `W.verify_completion`.
   Tiny synthetic проход всех 11 stages воспроизвёл `ValueError('completion
   fields differ')`. Нужно использовать `W.completion_identity` плюс `seconds`;
   adapter metadata оставить в отдельном operational receipt.

В том же synthetic проходе проверены семь удалений только новых intermediates,
сохранность четырёх final outputs и неизменность внешнего old-file sentinel.
Порядок last consumers сверён с STEPS и исходными producers. Staging использует
fresh WORK и copies, а не hardlinks; manifest/source/hash admission остаётся
reviewed generic implementation. Disk/RSS/wall являются monitored limits, не
измерениями peak/ETA и не kernel memory cap на Mac. Оценка 8.5GB не проверена
реальным build.

Измерений фактического build/runtime/RSS нет. Мониторинг раз в секунду может
реагировать после кратковременного превышения; Mac limits не являются cgroup
hard caps. Это соответствует ограниченному host delta и не меняет scientific
recipe. Full feature build для review не выполнялся.
