# Перезагрузка Mac: состояние PRC 7 октября 2026

Обновление14:45UTC: Mac worker27607 продолжает fold2, seed1000 завершён; memoryfree80%,13.3GB свободного диска, low counter0. Day controller33481 и E202 transport33495 запущены14:36UTC. Таблицы E202fold3/2 допущены14:41:31/14:42:33; Home GPU уже обучает flat seed0 epoch3/8. Fold5/6 передаются. E203 автоматически ждёт окончания Mac guard.175 отдельно завершился ошибкой отсутствующего sklearn после10 сохранённых bases; его recovery готовится без переобучения этих bases.171 fold1 завершён; Home/Hetzner продолжают CatBoost.

Снимок 14:20 UTC / 15:20 WEST. Пользователь сообщил о зависании и перезагрузке, затем прямо поручил снова запустить расчёты на Mac.

## Подтверждено

- Перезагрузился Mac примерно14:08UTC: новый launchd uptime и shutdown-stall report15:07:49WEST. Точная причина зависания не установлена. Отчёты ANECompilerService показывают отдельную системную CPU/дисковую активность; они не доказывают причинную связь с зависанием.
- Mac E201 fold1 завершился до перезагрузки,14:05UTC:344419строк,4548.55с, peakRSS18629705728bytes (~17.35GiB),8threads. CompleteSHA6a8cc558ec874206e0e2ff12cd06273ef71558da040dce23389dedc10ec324b3. Второйfold не начинался.
- Исходный lightweightaudit повторно выполнен после boot:22artifactSHA/recipe/probabilities/seedarithmetic PASS. TransferproofSHA38915251e1d492efd8df3064c1574565ddce00f0ae4c04cd8d165c3d3b03841e; smallpacket34,755,269bytes, SHA41235674c8b8cbad3d5293e71e0dc5c1360f7ad2da19f09e3938a3ee0676875c. Повторногоfit1нет.
- Home не перезагружался, uptime>2дня18ч; E201memoryfix PID627644 active, к14:20 завершеныLGBseed1000/1001fold3, current≈12.1GiB, peak16GiB. Старый failed12threadrun не возобновлялся. E202 PID626873active, но только ждётпервуюпровереннуютаблицу; это неGPUfit.
- Hetz не перезагружался, uptime>29дней; systemunitE201PID3897923active, Cat5600продолжается. Использовать systemctl без--user дляэтогоunit.
- Ноутбуки171/175 продолжаютE201 послеавтоматическогоrelease14:00:08/09UTC; готовыеE202таблицысохранены.
- Mac E202transportконтроллер и scpоборвались: brokenpipe/no route tohost. НаHomeсохранилисьнепроверенныеincomingfold2≈5.2GiB/fold3≈8.0GiB; готовымиinputsонинеявляются. Нужновосстановитьтолькопередачу, исходныеGPU/prepunitsнезапускатьзаново.
- SSHagentпослеbootпуст, восстановлен существующийid_ed25519 черезmacOSKeychain. Ключи/паролинеменялись.
- E203зарегистрирован№352/6e95e3a65936a3401e41499bb340f2d54a1a2121, НОникогданезапускался: отсутствуютcontrollerlog/state. Старыйpendingtoolприbootпотерялhostgeneration. Его нельзяописыватькакработающуюочередь.

## Возобновление

Пользователь15:17WEST требуетMacсчётснова. Плансдвигаfold2наHetzотменёнДОрегистрации/запуска. ОтдельныйтехническийMacwrapperпродолжаеттолькоfold2с6threads/nice10, контролемсистемнойпамяти/диска, noautoretry; scientificworker/inputsнеизменны. Первыйfoldнеповторять. E203localpack/applyдождутсяэтогоfold2, чтобыдватяжёлыхпроцессанеперекрывались.

E202transportвосстанавливаетсяотдельнымоперационнымпротоколом№354;Macвозобновление№353. ФактическиезапускиотражатьвexecutionnotesпослеPID/прогресса, непозаготовленномукоду.

## Подачи

Трипредыдущиеprocessedподачисохранены:12070=231.4449;12071=231.6095;12074=231.4438best.3/5UTCслотов7октября,двеостаются. Новыхподачпослерестартанебыло.

## Фактическое возобновление Mac

В14:28:39UTC запущены guardPID27601 и workerPID27607,6threads/nice10. Передстартом26.24GBreclaimableRAM/13.54GBfree disk/memoryfree89%; в14:30 memoryfree86%, lowercounter0. Workerначалinnerselectionfold2. Defaultпопытка былаостановлена os.nice PermissionError ДОchild/fit; state/log сохранены. Shellnohupпопытканесохранилаживойпроцесс; ниодногоfitона неуспела начать. УстойчивыйPopenstart_new_session черезpurpose-specificscripts/e201_mac_fold2_launch_20261007.py успешноразрешён; запуск проверенпоreceipts/логам.
