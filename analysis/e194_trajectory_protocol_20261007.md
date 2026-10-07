# E194 — реальные последовательности текущего и предыдущего ADS-B-визита

Код: `scripts/e194_trajectory_sequence.py`. Root владеет строкой журнала,
регистрационным коммитом и разрешением запуска. Настоящие target fit этим кодом
не предусмотрены. Никаких изменений production features/surfaces/visit-файлов,
существующих сред или чужих задач. Это новый безметочный input extraction для
будущего сравнения sequence/flat/reconstruction, не физические псевдометки E191.

## Механизм и границы прежних отрицательных результатов

Полезный новый вход — порядок реально наблюдаемых движений, остановок, поворотов,
скоростей, gaps и качества приёма собственного борта в текущем и предыдущем визитах.
E39/E134b проверяли обучение на искусственно обрезанных примерах: здесь нет ни
prefix augmentation, ни реконструкции скрытого начала как истинной метки.
E191 имел223 строгих physical-offblock наблюдения на187 днях со stale и177 дней с
неизвестным флагом; этот недостаточный auxiliary pilot не переиспользуется.
Отсутствующий stale сохраняется как UNKNOWN, не как False; airborne точки остаются.
Фактическая recorded TAXITIME и физическое движение не отождествляются.

## Реально доступные источники и population

На Mac `data/adsblol/surface_*.parquet`:364 дня2025 и31+31 день января/июля2026.
На31.12.2025 raw surface отсутствует: все departure ID этого дня всё равно
сохраняются, sequence полностью masked, status=sensor_missing. Иные неожиданные
пропуски surface останавливают pack. Поля t/lat/lon/gs/track/alt/on_ground/callsign
реальны; stale бывает отсутствующей колонкой и/или null.

Raw PRC: все12 `data/prc-2026-datasets/training_*.parquet` и `ranking.parquet`.
Проецируются ровно7 наблюдаемых полей: MVT_ID_mvt, PHASE_mvt, ADEP_mvt,
CALLSIGN_flt, FLIGHT_mvt, MVT_TIME_UTC_mvt, FLIGHT_ID_mvt. TAXITIME и BLOCK_TIME
не читаются. Разбиение строго по фактическому UTC дню MVT, а не имени месячного
файла: это сохраняет строки на двух известных пересекающихся границах файлов.
Проверены2 429 888 DEP ID, все уникальны; это2 085 047 train +344 841 ranking.
Всего427 дней =365 train +62 ranking. Feature/target/OOF/model файлы не нужны.

## Identity и границы визитов

Identity — ровно E64 same-day `(callsign,airport)->единственный ICAO`.
У departure CALLSIGN_flt с fallback FLIGHT_mvt удаляются пробелы, как в adsb_visit.
У points используются исходные callsign. Никаких расширений по похожим позывным,
flight number, вчерашним точкам или временной близости. Ambiguous/absent identity
даёт пустые последовательности с отдельным status, а не угадывание.

Для выбранного ICAO/airport точки сортируются по t и исходному row index.
Сохраняется E64 boundary policy: окно[MVT−43200,MVT+600], разрыв строго>1800с;
последний сегмент — current visit, непосредственно предыдущий — previous visit.
Сам набор токенов каждого сегмента дополнительно обрезается на **t<=MVT**.
Поэтому точки после MVT могут участвовать только в прежнем определении boundary,
но в tensor не попадают. Это явно retrospective boundary matching, не утверждение
полной online-causality. Ночная история из предыдущего файла не добавляется.

Это visit representation, не гарантированная непрерывная наземная траектория.
Смена позывного в предыдущем визите естественна; per-point совпадение с текущим
позывным хранится отдельно. Никакого запрета физически невозможных clock labels
из этих визитов не выводится. Неизвестные/несвежие точки остаются наблюдениями с
масками; право использовать их при будущем physics loss решается отдельно.

## Выход и отсутствие выдуманных наблюдений

Два потока × максимум64 выбранные **реальные** точки. При<=64 сохраняются все.
Иначе берётся uniform grid во времени визита, для каждого grid time выбирается
реальное первое наблюдение не раньше grid; первые/последние реальные точки
сохраняются, повторяющиеся индексы удаляются. Никакой интерполяции, усреднения
координат, синтетической неподвижности или нулевой скорости для PAD.

Numeric channels12: MVT−point time(seconds), gap от предыдущей исходной точки,
east/north(km) относительно фиксированного airport reference из adsb_extract,
altitude(ft), speed(kt), heading sin/cos, on_ground, stale, callsign-match,
duplicate_timestamp. `values` float32 и `known` uint8 на каждый channel;
`present` отдельно отмечает реальную точку. Missing value записывается нулём
только вместе с known=0, а padding имеет present=known=0. Ground=True с alt=null
не превращается в нулевую измеренную altitude. Stale UNKNOWN и known False различимы.

Каждый day output: sequence.npz, receipt.json. NPZ включает ID, ICAO identity,
число identity candidates/status, airport/MVT, values/known/present, исходный
source_row каждой выбранной точки, точное float64 point time, count всех точек
визита и полные raw_ranges в общей сортировке дня `(icao,airport,t,source_row)`.
**Все исходные точки доступны через immutable surface в bundle и эти диапазоны**:
64-point view не объявляется полным несжатым треком, будущий encoder может
восстановить весь сегмент без поиска identity заново. Timestamps-дубликаты
сохраняются детерминированно и помечаются; никаких производных ускорения не считают.

Receipts содержат SHA NPZ/кода, время дня, все ID count, status/support, наличие
stale schema, число наблюдений с известным/положительным stale, own/previous/both/
previous-only по дню и airport. Coverage не означает полезность/качество модели.

## Распределение на два ноутбука

Sorted427 days распределены по `DAYS[index::2]`: shard0=214дней, shard1=213дней.
Это непересекающиеся day shards, покрывающие оба года и все14 месяцев. Нет shared
outputs. На каждом ноутбуке workers3, Polars2, CPUQuota600%, MemoryMax10G,
MemorySwapMax0, nice10, existing env. Каждый worker держит только один день.
Существующие CI/другие jobs не останавливать. ASUS176 не используется.

Pack на Mac делает два **новых** bundle с копиями только surface своего shard,
маленькими projected departures по дням, script/protocol и SHA manifest. Source
files хэшируются до/после копирования; raw monthly hashes проверяются до/после
projection. Исполняемый script должен совпадать с bundle и регистрационным Git
объектом. Bundle переносится rsync, затем повторно проверяется на целевом узле.

Команды после root code+journal+protocol commit, SHA заменить полным hash:

```bash
.venv/bin/python scripts/e194_trajectory_sequence.py pack --root . --output data/e194_inputs_20261007_s0 --registration-commit SHA --shard-index 0 --shard-count 2
.venv/bin/python scripts/e194_trajectory_sequence.py pack --root . --output data/e194_inputs_20261007_s1 --registration-commit SHA --shard-index 1 --shard-count 2
rsync -a data/e194_inputs_20261007_s0/ a1@192.168.1.171:/home/a1/prc-2026/data/e194_inputs_20261007_s0/
rsync -a data/e194_inputs_20261007_s1/ a1@192.168.1.175:/home/a1/prc-2026/data/e194_inputs_20261007_s1/
```

171 запускает bundled script через существующую `.venv/bin/python`,175 через
существующую `.venv-cat-parity/bin/python` после imports/selftest. Пример171:

```bash
systemd-run --user --unit prc-e194-sequence-s0-20261007 --property=CPUQuota=600% --property=MemoryMax=10G --property=MemorySwapMax=0 --property=Nice=10 --property=RuntimeMaxSec=14400 --setenv=POLARS_MAX_THREADS=2 --working-directory=/home/a1/prc-2026 /home/a1/prc-2026/.venv/bin/python -u data/e194_inputs_20261007_s0/scripts/e194_trajectory_sequence.py run --bundle data/e194_inputs_20261007_s0 --output data/e194_sequence_20261007_s0 --registration-commit SHA --workers 3
```

Для175 заменить unit/suffix на s1, интерпретатор на `.venv-cat-parity/bin/python`.
Никаких установок пакетов или изменений среды. После запуска проверить живой
PID/CPU и появление дневных receipts; «передано» не означает «считается».
Сбой сохраняет partial/day artifacts; автоматического resume/ретрая нет.
complete.json публикуется только после всех дней shard и точного row coverage.

## Проверка и следующий эксперимент

До регистрации разрешены synthetic selftest:16 контрактов, disjoint shards,
строгая1800с граница, MVT cutoff, реальные source indices, missing sensor,
ambiguous identity, airport isolation, ground/stale unknown, duplicate ID rejection,
time-bin capacity, no-label column allowlist. Реальные source inventory/ID counts
измерены без чтения targets. Реальное извлечение — после регистрации root.

После полного run `verify --bundle ... --output ... --registration-commit SHA`
пересчитывает SHA/ID/mask/cutoff checks. Mac независимо проверяет оба complete,
точное покрытие427дней/2 429 888ID, отсутствие пересечения shard и source-row
соответствие выбранным фактическим raw points на фиксированном sample.

На этом шаге **accepted_for_training=false**: это не refusal future learning,
а точное отсутствие обучающей регистрации. Следующий отдельный screen сравнит
равноёмкие модели с одинаковыми selected raw tokens: flat pooling против
упорядоченного TCN/attention, с тем же FIN1b query и ordinary recorded residual.
Reconstruction/physics — отдельные arms после этого контроля. Нельзя трактовать
current/previous физическое время как новый истинный TAXITIME, импортировать
E191 pseudo-labels или вернуть провалившуюся prefix augmentation.

Будущая проверка учитывает естественное partial/previous-only покрытие, январь/
июль и LIRF, фиксированные holdouts/seed, body/trim после одинаковой цепочки.
Положительное support является только допуском сформулировать этот опыт, не
прогнозом RMSE. Дополнительные точки/большее число токенов не подбираются по OOF
после первых результатов без нового протокола.
