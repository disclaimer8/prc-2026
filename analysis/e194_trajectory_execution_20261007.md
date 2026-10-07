# E194 — извлечение завершено, все результаты проверены на Mac

Состояние на07.10.2026 после07:23Z: **427/427 дней, 2 429 888 уникальных DEP ID,
оба ноутбука закончили; обучение не запускалось**. Проверены214/1 217 396 строк
shard0 и213/1 212 492 строк shard1 штатным `verify` и независимым проходом.
Это готовые наблюдаемые входы для отдельного E197, не улучшение RMSE.

Регистрация: №332, `fc7107de145a87a5a1ff93314f318b440b1de825`.
SHA исполнявшегося `scripts/e194_trajectory_sequence.py`:
`c2cf0a747f25de4219680799e1dac0a162007258908306f9670e9c3474c2e0eb`.
Протокол: `analysis/e194_trajectory_protocol_20261007.md`.

## Реальное исполнение и durable receipts

| Узел | unit / последний рабочий MainPID | Дни / строки | Время extraction | systemd peak RAM | Завершение |
|---|---|---:|---:|---:|---|
|a1@192.168.1.171|prc-e194-sequence-s0-20261007 /255802|214 /1 217 396|137,935с|4,4GiB|07:22:27Z|
|a1@192.168.1.175|prc-e194-sequence-s1-20261007 /89662|213 /1 212 492|165,546с|3,5GiB|07:23:11Z|

На каждом: workers3, Polars2, CPUQuota600%, MemoryMax10G, MemorySwapMax0,
nice10, RuntimeMaxSec14400, sleep/idle inhibitor.171 использовал `.venv/bin/python`,
175 `.venv-cat-parity/bin/python`; обе существующие среды: Python3.14.4,
NumPy2.5.3, Polars1.44.2. Среды не менялись. Полная длительность units с проверкой
входов:140,376с и169,016с; CPU508,434с и614,505с.
Последний `systemctl show` после завершения:inactive/dead, MainPID0, ExecMainStatus0.
Transient units уже выгружены, поэтому их текущие default MemoryMax/CPUQuota
не являются значениями реально отработавшей задачи; ресурсы подтверждались при запуске.

Логи: `journalctl --user -u prc-e194-sequence-s0-20261007.service` на171 и
аналогичный`s1` на175. Working directory обоих `/home/a1/prc-2026`.
Полные входы/выходы сохранены на ноутбуках и Mac:

- `data/e194_inputs_20261007_s0`, `data/e194_inputs_20261007_s1`;
- `data/e194_sequence_20261007_s0`, `data/e194_sequence_20261007_s1`;
- каждый `complete.json` содержит все дневные receipts, runtime и регистрацию;
- `data/e194_verification_20261007/verification.json` — независимая итоговая проверка;
- `data/e194_verification_20261007/verify_collected.py` — её воспроизводимый код.

Input manifest SHA s0:`a2dc178a960f969d137dee4989610922fc4a2da787279118f3aae9cd93f4a4c8`;
s1:`49a62a28bc2b8fe689aaa1581838167c6ad42e2b0e22a0363e8ec4308ad2fbdd`.

## Команды и проверка

Pack выполнен на Mac, `SHA` ниже — полный регистрационный commit:

```bash
SHA=fc7107de145a87a5a1ff93314f318b440b1de825
.venv/bin/python scripts/e194_trajectory_sequence.py pack --root . --output data/e194_inputs_20261007_s0 --registration-commit "$SHA" --shard-index 0 --shard-count 2
.venv/bin/python scripts/e194_trajectory_sequence.py pack --root . --output data/e194_inputs_20261007_s1 --registration-commit "$SHA" --shard-index 1 --shard-count 2
rsync -a data/e194_inputs_20261007_s0/ a1@192.168.1.171:/home/a1/prc-2026/data/e194_inputs_20261007_s0/
rsync -a data/e194_inputs_20261007_s1/ a1@192.168.1.175:/home/a1/prc-2026/data/e194_inputs_20261007_s1/
```

Payload запусков — следующие команды из рабочего каталога ноутбука, обёрнутые
описанным выше systemd-run и `systemd-inhibit --what=sleep:idle --mode=block --who=PRC-E194 --why=E194-raw-sequences`:

```bash
.venv/bin/python -u data/e194_inputs_20261007_s0/scripts/e194_trajectory_sequence.py run --bundle data/e194_inputs_20261007_s0 --output data/e194_sequence_20261007_s0 --registration-commit fc7107de145a87a5a1ff93314f318b440b1de825 --workers 3
.venv-cat-parity/bin/python -u data/e194_inputs_20261007_s1/scripts/e194_trajectory_sequence.py run --bundle data/e194_inputs_20261007_s1 --output data/e194_sequence_20261007_s1 --registration-commit fc7107de145a87a5a1ff93314f318b440b1de825 --workers 3
```

Коллекция и фактически выполненная проверка на Mac:

```bash
rsync -a a1@192.168.1.171:/home/a1/prc-2026/data/e194_sequence_20261007_s0/ data/e194_sequence_20261007_s0/
rsync -a a1@192.168.1.175:/home/a1/prc-2026/data/e194_sequence_20261007_s1/ data/e194_sequence_20261007_s1/
.venv/bin/python data/e194_inputs_20261007_s0/scripts/e194_trajectory_sequence.py verify --bundle data/e194_inputs_20261007_s0 --output data/e194_sequence_20261007_s0 --registration-commit fc7107de145a87a5a1ff93314f318b440b1de825
.venv/bin/python data/e194_inputs_20261007_s1/scripts/e194_trajectory_sequence.py verify --bundle data/e194_inputs_20261007_s1 --output data/e194_sequence_20261007_s1 --registration-commit fc7107de145a87a5a1ff93314f318b440b1de825
.venv/bin/python data/e194_verification_20261007/verify_collected.py
```

Все три проверки завершились exit0/PASS. Точные ID по дням/объединению,
входные/выходные SHA, finite values, known≤present, отсутствие token t>MVT,
порядок timestamps, PADsource=-1/PADtime=NaN, непротиворечивость raw_ranges/counts,
дневных receipts и общего complete подтверждены. Две части не пересекаются.
На фиксированных01.01.2025/01.07.2025/01.01.2026/01.07.2026 выборках по аэропортам
дополнительно проверены реальные source rows, ICAO/airport/time, speed/altitude,
ground/stale known и значения, координаты и возраст точки.31.12.2025 сохранены
все4715 рейсов с полностью пустыми масками: отсутствующий sensor не удаляет строки.

Первый реальный день01.01.2025 отдельно обработан на Mac за0,585с:4788 рейсов,
2307 own/2798 previous/1021 previous-only,154669 выбранных точек.
Сравнение с Linux по всем массивам PASS; точная величина возможного float32
различия записана в `mac_linux_first_day_parity` итогового JSON. Целочисленные
ID/индексы, времена и маски обязаны совпадать точно.

## Измеренное покрытие, без вывода о качестве модели

| Population | DEP | Current visit | Previous visit | Previous only | Настоящих selected points |
|---|---:|---:|---:|---:|---:|
|2025 train|2 085 047|1 251 417|1 345 381|322 974|88 770 275|
|2026 January ranking|152 719|105 052|100 834|16 213|6 999 686|
|2026 July ranking|192 122|122 645|132 153|30 607|8 754 076|
|Всего|2 429 888|1 479 114|1 578 368|369 794|104 524 037|

Оба визита представлены у1 208 574 строк. Неизвестный stale имеется у45 866 940
выбранных точек; известен у58 657 097, из них8 114 349 помечены stale.
На ranking stale schema доступна для всех выбранных точек; train имеет прежний
177-дневный пробел. Подробные airport/period counts находятся в verification.json.
Покрытие current ниже некоторых старых visit-counts, потому что после выбора
границ токены строго обрезаны наMVT. Сам boundary policy всё ещё может смотреть
наточки доMVT+600, как E64: online causality не заявляется.

## Следующий гейт и ограничения

`accepted_for_training=false` в E194 означает, что он извлекает данные и не
регистрирует target-fit. Следующий E197 — отдельный протокол/коммит: одинаковые
immutable E190 query tensors, одинаковые2×64 actual tokens/known/present;
равноёмкие flat summary и local ordered mixing. В обоих arms остаются ages/gaps:
проверяется польза локального порядка, а не наличие всей временной информации.
Нормализация только по train fold; пары1+7 и4+10,3seed/8epoch; контроль body,
decided и trim относительно matched flat и сохранённого PLE. Против PLE эффект
смешивает новые входы с дополнительной ёмкостью, поэтому чистую атрибуцию входам
не заявлять. Физические псевдометки, prefix augmentation и reconstruction не входят.

При нарушении SHA/ID/masks дальнейший fit запрещён до новой проверяемой версии.
Нет автоматического resume/замены production surfaces. Возможные ограничения:
callsign ambiguity, same-day история без ночного продолжения, неизвестный stale,
обрезанное наблюдение, sparse LTFM, retrospective boundary. Наличие последовательности
не доказывает физический off-block или совпадение recorded TAXITIME с движением.
