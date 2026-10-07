# E202 — восстановление передачи после перезагрузки Mac

Операционныйпротокол `E202-postreboot-transport-only-original-gpu-no-refit-v1`.
Worker `scripts/e202_postreboot_transport_20261007.py`. Root регистрирует354
и запускает в общем purpose-specific daylauncher послеreview.

Перезагрузка Mac прервала исходный scp-3 transport. Четыре исходные CPUтаблицы
на171/175 готовы и сохранены; ноутбуки уже продолжаютE201. Исходный Home GPUunit
`prc-e202-trajectory-20261007.service`, PID626873, invocation
`30d9322da32d4cfdb55ef5c7c3d5ecbf`, ожидает admittedfold2. Новые GPU/prep units
не запускаются, fit/preparation не повторяются, scientificE202/E197 не меняются.
Первыйfold должен быть admitted до исходного двухчасовогоdeadline≈15:57UTC.

Неизменные registration `55fe9b68f7cc8405681f6c85d576979e067b55c3`, spec
`6d604eaaca830fa83c18ebb78986b028d338869123d5f8c10b2450dde1ec4309`.
НаMac проверяются зарегистрированныеsource/manifest/queries; наHome manifest;
накаждомноутбуке exact two-prepared marker, completeSHA, sourceSHA и6fileinventory.
Фиксированное распределение171fold2→5 и175fold3→6, двапараллельныхstream.

Для каждого файлаincomingHome сначала проверяетсяSHA. Верныйфайл не копируется.
Неполныйилиневерный existingfile перемещается безперезаписи в
`data/e202_postreboot_archive_20261007/fold_N_M/`; существующийархивконфликт
останавливает запуск. Далее checksumисходного immutablefile, fresh scp-3 и
checksumHome. Нет удаления/перезаписи сохранённыхpartials. Ожиданиепередачи
не блокирует E201 notebookfit, путираздельные.

Полностьюпереданнаятаблица проходит **исходный Q.admit** в отдельномnamedunit
`prc-e202-postreboot-admit-fN-20261007.service`,1CPU/1GiB/swap0/nice10/1800s.
Одновременно толькоодноadmission. Q проверяет ID/mapping/normalization/SHA/
shapes/masks/support/observations иатомарнопереноситincoming→prepared.
Существующийvalidadmissionсохраняется. OriginalGPU увидитготовуюпару сам;
invocationдолжнасохраняться, failure/исчезновение безcomplete закрываетrecovery.

Послеисходных24fit иterminalsuccess: компактныйmodel/result output копируется
наMac, complete/evaluationSHA сверяютсясHome. Новая hardlinkedreviewкопия
исполняет прежний Q.evaluate; exactJSONравенство обязательно. Большиеprepared
таблицы остаютсянаузлах;Mac37GiBнепринимает. Нетranking/upload.

State/log: `data/e202_postreboot_transport_20261007.json` и`.jsonl`.
Fresh one-shot recovery; ошибкисети/хеша/admission/GPU→failure, noautoretry.
Внешний rootdaylauncher обеспечиваетразрешённыйсетевойконтекст. Агент не
запрашиваетновоеgenericapprovalи не запускаеткопиюисходногоGPUпроцесса.
