# E197 — исполнение зарегистрированного сравнения последовательностей

Состояние на07.10.2026 09:06Z: **CPU preparation и Home admission завершены**,
обе большие таблицы проверены и опубликованы наHome. GPU очередь E197 уже
ACTIVE и наблюдает работающийE198; сами12fit E197 ещё не запускались.
Это operational note, frozen protocol/code
после регистрации не менялись.

Регистрация E197№336:`bfe793c62f7cecf633c7cc31c747f48eb0676f2d`.
Протокол:`analysis/e197_trajectory_protocol_20261007.md`.
Input bundle:`data/e197_inputs_20261007` (156MiB), подготовлен наMac и доставлен
на171/175/Home. В нём365raw-day SHA, exact immutable E190/E192 mapping иcontrols.

## Реальная подготовка

|Узел|unit / рабочийPID|Fold|Строк|Preparation|ЗавершениеUTC|
|---|---|---|---:|---:|---|
|171|prc-e197-prep-f1-20261007 /263799|1+7|2 085 047|66,203с|08:51:13|
|175|prc-e197-prep-f4-20261007 /96137|4+10|2 085 047|56,682с|08:51:05|

Оба complete опубликованы, units inactive/dead. Общий systemd wall включаяSHA:
75,421с/67,844с, CPU142,717с/147,150с. Пикmemory.current≈10GiB на каждом
(включает mmap/file cache, не только Python heap), ограничение MemoryMax10G
соблюдено; OOM/падений нет. По3worker thread,CPU600%,swap0,nice10,4h watchdog,
sleep/idle inhibitor.171 `.venv`,175 `.venv-cat-parity`; среды не менялись.
Логи:`journalctl --user -u UNIT.service` на соответствующем ноутбуке.

Small durable receipts наMac:
`data/e197_preparation_receipts_20261007/fold_{1_7,4_10}/{complete,normalization}.json`.
Оригиналы large tables:`/home/a1/prc-2026/data/e197_prepared_20261007/fold_*`
на соответствующем ноутбуке. Обычных trainquery1 622 159/1 607 392, heldout
344 419/360 962; normalization обучена только наtrainindices, исключениеheldout
месяцев проверялось по каждому исходному дню. Known observationsдляage/gap/x/y:
72 184 574/69 662 105, altitude15 113 409/14 770 408, speed72 045 138/69 527 559.
Никаких ranking observations в normalization нет.

Точныеpayload-команды послеpack иrsync (cwd `/home/a1/prc-2026`):

```bash
.venv/bin/python -u data/e197_inputs_20261007/scripts/e197_trajectory_encoder.py prepare-fold --root /home/a1/prc-2026 --input data/e197_inputs_20261007 --output data/e197_prepared_20261007/fold_1_7 --fold 1 --registration-commit bfe793c62f7cecf633c7cc31c747f48eb0676f2d
.venv-cat-parity/bin/python -u data/e197_inputs_20261007/scripts/e197_trajectory_encoder.py prepare-fold --root /home/a1/prc-2026 --input data/e197_inputs_20261007 --output data/e197_prepared_20261007/fold_4_10 --fold 4 --registration-commit bfe793c62f7cecf633c7cc31c747f48eb0676f2d
```

Обе команды фактически обёрнуты `systemd-run --user` с указаннымиguards и
`systemd-inhibit --what=sleep:idle --mode=block --who=PRC-E197 --why=E197-sequence-preparation`.

## Проверенный GPU smoke

Home `.venv-numemb`, тот жеGPU runtime, что immutable E190query tensors:
сравнение runtime наMac PASS. Service`prc-e197-smoke-20261007` завершился
08:47:24Z, wall6,095с, RAM1,1GiB. Модель обучалась только наsyntheticarrays.
GPUreserve равен полномуtoken-table объёму9,2GiB, плюсfull batch4096.

|Arm|Synthetic parameters|Времяbatch|Peak GPU bytes|
|---|---:|---:|---:|
|flat|2 452 821|0,076825с|13 024 943 104|
|ordered|2 452 821|0,073084с|12 751 250 432|

Это число параметров **synthetic query shape**, не фактическийcount реальных
фолдов. Equal capacity, finite gradients, PAD/unknown invariance, flat
permutation invariance, ordered adjacency sensitivity, no cross-visit mixing
прошли. Receipt`data/e197_smoke_20261007.json` наHome иMac; sourceSHA совпадает.
Ориентир≈48мин чистого12fit без подготовки/проверок, не обещание времени.

## Перенос и следующая очередь

НаMac всего≈9GiB свободно, поэтому **large prepared tables наMac не собираются**.
Ноутбуки ещё не имеют доверенногоSSHhostkey дляHome; новые ключи/agent forwarding
не настраивались. Используется `scp -3` через существующие дваSSH-доступаMac:
поток notebook→Home без промежуточного файла наMac.

```bash
scp -3 -r -o BatchMode=yes a1@192.168.1.171:/home/a1/prc-2026/data/e197_prepared_20261007/fold_1_7 home-pc-lan:/home/a1/prc-2026/data/e197_transfer_20261007/
scp -3 -r -o BatchMode=yes a1@192.168.1.175:/home/a1/prc-2026/data/e197_prepared_20261007/fold_4_10 home-pc-lan:/home/a1/prc-2026/data/e197_transfer_20261007/
```

Обе передачи завершились exit0. Home admission завершён09:04:47Z, обаfoldPASS:
SHA/lineage всехpreparedfiles, все2 085 047IDs, shape/dtype,finite,known/PAD,
clip±8 иbinary/heading ranges, contiguous tokens/support. На каждомfold точно
88 770 275selectedpoints,42 903 335knownstale,1 251 417own/1 345 381previous/
1 022 407both — совпадает сE194 train. Проверенныеdirectories опубликованы
atomic rename в`data/e197_prepared_20261007`. Largefiles наMac не создавались.
Admissionкод:`data/e197_queue_assets_20261007/admit_prepared.py`; smallreceipts
наMac:`data/e197_preparation_receipts_20261007/e197_admission_20261007_f{1,4}.json`.

InputspecSHA:`93d90131cdfa32c04b8f8c3513f3e7f7cc19c9695d8f47d3fe98b4cc8c0a407d`.
WorkerSHA:`6be13d401dd2903c50c9584f5ca2d6ca084a1feb0eb1f67fcba0cd94850db54a`.
PreparedcompleteSHA f1:`cdd45087b8aef0af1dfb8e8eec8b4c586feae66194d9afe60d876375f486f8c8`;
f4:`835e213657e6703cf835b98c6644c04bd6723f3c722ea7a2233f16580634be05`.

Queuewrapper:`data/e197_queue_assets_20261007/wait_then_run.py`, отдельно от
статистического registeredworker. E198 precursor:
`prc-e198-density-20261007.service`, registration
`25ef25db84657bccefa48be7ac42c76c873520e3`,
`data/e198_density_inputs_20261007/manifest.json`,
`data/e198_density_20261007/complete.json`.
Нужно terminal inactive/MainPID0 и correctregistration/spec/8arms×3seeds×8epochs;
отсутствиеunit не считается завершением. Failure/исчезновениепослестарта без
complete останавливаеточередь. Максимумwait6h, fit4h, безauto-retry.
E197 запустит `verify`, затемregistered12fit, затем`evaluate`; автоматических
подач/расширенийнет. Durable timeline:`data/e197_queue_20261007.jsonl` наHome.
Precursor specSHA закреплён как
`37a13dbb83e852c5ef4a6bde9b84dfd9a2882affc5fc44d8c2654658602c691f`;
queue проверяет его model/prediction hashes иevaluationreceipt, не требует
положительногоE198qualitygate для независимой гипотезыE197.

**Фактическая очередь запущена09:06:13Z**, unit
`prc-e197-trajectory-20261007.service`, MainPID**591693**, invocation
`0b48f00de65a4f1daada3b2fdcd094cc`, Active/running.
Она уже наблюдала E198 Active/MainPID589259 с exactinvocation
`99f7d510c8dd4ccbbeecae5ac6f9fbac`; отсутствиеunit доегостарта не выдается зауспех.
Homeguard12GiB/swap0/CPU200%/nice10, общийqueueRuntimeMax41400с с отдельным
fit-timeout4h. `.venv-numemb` сохранена. Точная команда запуска:

```bash
systemd-run --user --unit=prc-e197-trajectory-20261007 --property=MemoryMax=12G --property=MemorySwapMax=0 --property=CPUQuota=200% --property=Nice=10 --property=RuntimeMaxSec=41400 --property=WorkingDirectory=/home/a1/prc-2026 /home/a1/prc-2026/.venv-numemb/bin/python -u data/e197_wait_queue_20261007.py --predecessor-registration 25ef25db84657bccefa48be7ac42c76c873520e3
```

Мониторинг:`journalctl --user -u prc-e197-trajectory-20261007.service`,
`data/e197_queue_20261007.jsonl`; конечныйрезультат
`data/e197_trajectory_20261007/{complete,evaluation}.json`.
До `launch_e197_fit` и первойэпохи нельзя утверждать, что модель уже обучается.
