# E201 — Macfold2 после перезагрузки, с ограничением ресурсов

Пользователь прямо попросил снова запускать расчёты на Mac. Ранее предложенный
переносfold2наHetz отменён до регистрации и до запуска. Все узлы продолжают
свои прежние задачи; нет дополнительногоHetzfit.

Macfold1 завершился14:05UTC перед rebootоколо14:08. Его22артефакта проверены
root original-node audit:complete SHA
`6a8cc558ec874206e0e2ff12cd06273ef71558da040dce23389dedc10ec324b3`,
audit SHA `38915251e1d492efd8df3064c1574565ddce00f0ae4c04cd8d165c3d3b03841e`.
Smallpacket34,755,269bytes уже опубликован, SHA
`41235674c8b8cbad3d5293e71e0dc5c1360f7ad2da19f09e3938a3ee0676875c`.
Fold2 и егоpartial отсутствуют; ранее эта пара не обучалась. Первыйfold
сохраняется и не повторяется. Причина перезагрузки не установлена.

`scripts/e201_mac_fold2_guard_20261007.py` выполняет только v60fold2 снуля,
используя исходный frozenworker SHA
`9da861964ddec3dade4c2a1438c20214ee21236388b9337b3fffb16cab594cbd`,
registration `d664e153b3f0732b4047011cc062f58f42581a6c`, manifest
`f8696b7326f196ba72f51e44be5a6b658c8f063fcac09e1bd54274c80218a2aa`.
Scientificrecipe/седы/innerroundselection/цели/входы неизменны.
Операционное число потоков8→6,nice10; runtime тотжеMac `.venv`.
Ни resume partialfit, ни autoretry, ни подбора по ошибкам.

Перед запуском:≥16GiB reclaimable RAM поvm_stat(free+inactive+speculative),
≥4GiB free disk; провереныsource/input/firstfoldSHA и отсутствиеfold2/partial.
Во время только собственногоchild: каждые10сек `memory_pressure -Q`;
если freepercentage<10%триразаподряд, либоdiskfree<4GiB, либо48hdeadline,
процесснаягруппа собственногоchild получаетTERM,30секgrace, затемKILL
толькопри необходимости. Никакие другиепроцессы не трогаются. Глобальныйps
не вызывается; RSSлимит напрямую не заявлен. Давлениепамяти и disk записываются
вJSONL. Startupилиruntimeошибка даётfailure/сохранённыйpartial, не retry.

`data/e201_mac_fold2_guard_20261007` — отдельные operationalreceipts.
Результат иsmallpacket остаются в исходном `data/e201_mac_20261007/v60_fold2`,
поэтому обычныйcollector сохраняет Macfold1/2mapping; Home выбирается
явным`--home-memoryfix`. Послеfit originalaudit проверяетP/сидовуюарифметику/
всеSHA; только затемfold2_ready. E203daycontroller ждёт этогоsuccess ДО
любыхlocalCV/pack, чтобы не складывать тяжелыеMacзадачи.

Root запускает CPU-onlywrapper подcaffeinate default sandbox послеreview и
регистрации353. Никакихсетевыхвызовов/новыхenvironments/GPU/submission здесьнет.
Состояние до запуска:compile+memory-onlyprobe, без новыхtargetfits.
