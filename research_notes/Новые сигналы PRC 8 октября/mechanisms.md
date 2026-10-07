# Новые локальные механизмы PRC: неиспользованные токены и отдельная задача на прилётах

## 1. Что действительно теряется в текущих 2×64 точках?

### Takeaway

**Первый кандидат — заполнить уже выделенные, но пустые token slots настоящими наблюдениями.** Обнаружена массовая потеря входа: при наличии >64 raw points текущий sampler часто оставляет около40. Это конкретная новая проверка входов, отличная от изменения свёртки E204. Второй контраст проверяет, полезнее ли заполнять slots переходами скорости, чем только временным покрытием. [Код sampler](../../scripts/e194_trajectory_sequence.py:177), [готовый draft-протокол](mechanism_sampler_protocol.md).

### Cited Findings

- Точная реализация:64 равномерных **временных** позиций →`searchsorted` →`unique`. При нерегулярности несколько grid positions попадают в один raw index. Дополнительного заполнения нет. E197 применяет два локальных message blocks, mean/max pooling и presence fraction. `gap` относится к предыдущей RAW точке; это семантика, не баг, а времена выбранных точек уже доступны через age. [Sampler](../../scripts/e194_trajectory_sequence.py:177), [каналы](../../scripts/e194_trajectory_sequence.py:288), [encoder](../../scripts/e197_trajectory_encoder.py:288).
- Полный новый **metadata-only** проход427 E194 NPZ занял3.09с; читались ids/airport/raw_counts/point_times/takeoff, без targets/BLOCK/OOF. Выходы/код сохранены; E194 раньше независимо сверял все ID/источники. В этой новой metadata-пробе повторное хеширование всех больших NPZ не выполнялось. [Helper](mechanism_sampler_inventory.py), [JSON](mechanism_sampler_inventory.json), [предыдущая полная проверка E194](../../analysis/e194_trajectory_execution_20261007.md).

| Период/stream | Все DEP | Stream есть | Raw>64 | % всех DEP / % с этим stream | Из raw>64: selected<64 | Медиана selected при raw>64 |
|---|---:|---:|---:|---:|---:|---:|
|2025 own|2 085 047|1 251 417|666 621|31.97% /53.27%|663 305|42|
|2025 previous|2 085 047|1 345 381|709 083|34.01% /52.70%|704 168|41|
|Jan2026 own|152 719|105 052|55 296|36.21% /52.64%|54 938|43|
|Jan2026 previous|152 719|100 834|75 957|49.74% /75.33%|75 153|38|
|Jul2026 own|192 122|122 645|64 403|33.52% /52.51%|63 943|44|
|Jul2026 previous|192 122|132 153|96 449|50.20% /72.98%|95 855|39|

Источник всех строк таблицы: [полный metadata JSON](mechanism_sampler_inventory.json). Own/previous множества пересекаются: эти числа нельзя суммировать как число затронутых рейсов. Покрытие E194 отличается от старых `own_track` masks Fable из-за определения визита/точек; не заменять одно другим.

- Проверка четырёх **заранее фиксированных календарных дат**, тех же, что использовала независимая проверка E194:01Jan/01Jul2025/2026. На own-stream с raw>64 определено только наблюдаемое пересечение1kt снизу: finite gs, соседние raw points, gap∈(0,30]с. Из1123/1722/1828/1767 таких compressed visits у473/744/783/751 есть хотя бы одна crossing pair, оба конца которой не сохранены вместе. Ни один endpoint не попал в selected у98/131/181/151 crossing pairs. Это **не** доказанные off-block события, не все onset и не репрезентативная оценка на весь год. [Helper/определение](mechanism_transition_metadata.py), [результат](mechanism_transition_metadata.json).
- Начальная история «длинная стоянка съела весь бюджет» оказалась узкой: own span>1h всего16 191train/3 142ranking;>2h2340/546. Главный масштаб — нерегулярная дискретизация даже обычных коротких визитов. Стояночная длительность здесь не измерялась, span не равен stationarytime. [Metadata](mechanism_sampler_inventory.json).
- E197/E202 с новым сырым входом прошёл текущий preE81 через E203:body−0.614049с/decided−0.545761,6/6; полный v12075 улучшил доску на0.5051с. E204 оставил **те же selected points** и изменил второй message offset±1→±8; FAIL двухпарного гейта. Ни один не проверял дополнение самих токенов. [E203](../../analysis/e203_upload_20261007/RESULTS.md), [E204](../../analysis/night_gpu_20261007/RESULTS.md), [журнал№336/350/369/375](../../HYPOTHESES_LOG.md).

### Inferences

- Это лучший следующий опыт по ожидаемой **информационной ценности**: изменяется конкретно отсутствовавшее наблюдение при том же числе параметров. Затронутых ranking потоков много, сборка входа уже существует. Известные E25/E63/E69/E70/E74/E141 дают scalar onset/stop/stand-walk summaries; новый предмет — сохранённый локальный контекст перехода и соседних точек, а не ещё одна firstmove-поправка. Сами scalar summaries могут полностью поглотить пользу. [Текущая архитектура](../../docs/model-design.md), [журнал№183/362](../../HYPOTHESES_LOG.md).
- Root выбрал до любых новых target evaluations **все6пар сразу:18 новых fill-only fits,6×3seeds×8epochs**, saved ordered без повторного control fit. Модель/каналы/вес.25фиксированы; O(n)evenly-spacedunselectedraw-indexfiller. Гейт≥4/6decided, pooledbody/decided/within-foldtrim3/20<0, затемотдельныйcurrentE203chain. TransitionfillT−F/T−old толькоотдельныйfollow-up. HomeGPU:≈72–75мин training поE204baseline; меньшиевариантыF6fit≈24мин/F+T12fit≈48мин были ресурснымиопциями, невыбранныйпротокол. [Полный draft](mechanism_sampler_protocol.md), [минимальныйadapter/storage](mechanism_implementation_assessment.md), [E204 runtime](../../analysis/night_gpu_20261007/RESULTS.md).

- Контрпример: свободные slots заполняются в основном stale/duplicate rows, а новые точки меняют pooling и локальную смежность к худшему. Eventfill дополнительно может сохранить jitter около1kt. Поэтому **same capacity ≠ same information**, и matched fill-only контроль обязателен. [Формулы encoder](../../scripts/e197_trajectory_encoder.py:296).

### Gaps

- Ни качество этих новых samplers, ни условная ошибка текущего E203 на raw>64 не измерялись. Новый target probe не запускался. Нельзя переводить119699 compressed own ranking streams в обещание секунд.
- Четыре дня показывают реальную потерю некоторых переходов, но ещё не переносимость на весь январь/июль и не физическую корректность этих переходов. Статус stale не использован в определении crossing; он остаётся известным входом/ограничением.
- Читалась primary method часть Neural CDE о continuous path/irregular sampling, intensity/missing channels. Это подтверждает, что representation нерегулярного потока — отдельный выбор, но **не** свидетельство эффективности CDE у нас; новый solver не предлагается до простого sampler-контраста. [Kidger et al., methods§3](https://arxiv.org/html/2005.08926v2).

## 2. Можно ли обучить encoder отдельным наблюдаемым endpoint прилёта?

### Takeaway

**Резерв после sampler: shared trajectory encoder с отдельной ARR-задачей предсказания записанного in-block момента.** Это новая индивидуальная обучающая задача на другом типе движения; её новизна относительно журнала есть, но доказанной пользы и даже raw-track coverage для ARR пока нет. [Отличие от существующего DEP MSE encoder](../../scripts/e197_trajectory_encoder.py), [правила данных](https://prc-data-challenge-2026.netlify.app/data.html).

### Cited Findings

- Официальная страница явно говорит, что BLOCK/TAXITIME скрыты у **DEP**, а ARR MVT=landing и ARR BLOCK=in-block. В ranking локально344693ARR; у всех344693BLOCKnonnull,342141CALLSIGNnonnull,344688STANDnonnull. Это только null-count metadata, не оценка по arrival target. [Официальные определения/доступность](https://prc-data-challenge-2026.netlify.app/data.html), [receipt команд/хешей](mechanism_evidence_receipt.json).
- E194 извлекает только DEP и отсекает точки t>takeoff; применять его неизменённым к ARR нельзя: in-block обычно после landing. E197 учится по DEP target, а previous visit не получает отдельной endpoint supervision. [E194 DEP_COLS/pack](../../scripts/e194_trajectory_sequence.py:26), [E197](../../scripts/e197_trajectory_encoder.py).
- H15/H17/E162 добавляли inbound delay/slack/age как **признаки** и не обучали encoder на индивидуальном ARR BLOCK; E162 отклонён decided+0.016,3/6. №187 проверял taxi-in airport×month calibration, корреляция+0.057; FableC P2/P3/P6 — arrival recording-rate/day-clock/day-level probes. №326 — reconstruction/physics на собственном треке; E191 — восстановление sparse flags с остановленным масштабированием. Всё это близкие отрицательные/открытые направления, но не proposed shared ARR endpoint loss. [Журнал№180/182/196/187/307/326](../../HYPOTHESES_LOG.md), [FableC§4.2](../../analysis/review_fable_2026-10-07/fable_C_error_budget.md).
- Уже локальная Yang2024§3.8 использует отдельные auxiliary flow targets, суммируя task losses. Это методический пример, не наша новая статья и не доказательство переноса ARR→DEP. Прочитаны local methods§3.5–3.8; publisher fetch вернул ошибку, локальный полный текст сохранён ранее. [Primary DOI](https://doi.org/10.3390/aerospace11050371), [локальный текст](../../analysis/literature/MDPI_Yang2024_MT_DSTGAN_PEK.txt). Sener/Koltun формализуют shared/task-specific parameters и конфликт objectives; сложное adaptive weighting в первый опыт не нужно. [Primary methods§3](https://papers.nips.cc/paper_files/paper/2018/file/432aca3a1e345e339f35a30c8f65edce-Paper.pdf).

### Inferences

- Носитель возможной пользы — выученное различение движения/остановки/подхода к стоянке при изменении доступности датчика. Это прежде всего те же own/previous rows из раздела1. Нового наблюдения для LTFM без ground coverage не появляется. Старые FableC projected-error ячейки LFPG previous иLIRF previous мотивируют проверку, но относятся к E195, не текущему E203; опубликованные там «полы» не доказывают математическую неснижаемость ошибки. [FableC§2–3 и оговорки](../../analysis/review_fable_2026-10-07/fable_C_error_budget.md).
- Первый шаг только metadata: для ARR уникальный callsign/airport identity, фиксированное окно [landing−120с, landing+7200с],≥2 actualpoints, known speed/position/masks, включая переход черезUTC midnight; counts отдельно airport/month2025 иJan/Jul2026. Никакого crop по BLOCK. Иначе задачка становится утечкой endpoint через длину trace. Только после этого — root protocol+commit.
- Минимальный supervised pilot: shared token/message weights, отдельный ARR head предсказывает **recorded** BLOCK−landing, DEP head/выход/данные неизменны. Сравнить одинаковые архитектуры/DEP updates с λARR=0 и фиксированным λARR>0; ARR label не передавать в token/query inputs или crop. Auxiliary targets масштабировать только fit-month statistics, один заранее заданный вес, без сетки. На первом пилоте использовать только ARR fit-month2025; holdout-month ARR и2026ARR для обучения не использовать, чтобы не смешать эффект multitask с трансдуктивной адаптацией.
- Falsification:≤12fits(2 arms×2 пары×3seeds), same original K64 sampler; ровно те же body/decided/trim criteria, затемremaining4/currentE203chain приPASS. Сначала HomeGPU synthetic memory smoke;2CPU/12GiB RAM/20GiB VRAM и≤6hpilot. Длительность не измерена, ARRcoverage неизвестно. Простая zero-auxiliary ablation должна победить любую историю про пользуtask loss, если DEPметрика хуже.
- Контрпример1: ARR recordedBLOCK подменяется расписанием/другими часами; физический parking state не идентифицирует label. Контрпример2: ARR тормозит кстоянке, DEP pushback имеет иные скорости/heading/reception; перенос отрицателен. Поэтому этот кандидат ниже информационной потери sampler, прогноз RMSE неизвестен.

### Gaps

- ARR track linking/coverage в нужном post-landing окне не измерены, value этой дополнительной supervision неизвестна. Это резервный протокол, не готовая очередь и не promise наслепыеLTFMрейсы.
- Позднее применение разрешённых2026ARR labels как transductive training требует отдельного протокола и time-matched CV, в котором доступны ARR holdout месяца, но недоступны DEP labels. Смешивать это с первым pilot нельзя.

## 3. Что не выдавать за третью новую идею и где граница доказательств?

### Takeaway

Рекомендованы две линии с разным уровнем готовности: **sampler — реализовать и зарегистрировать первым; ARR endpoint auxiliary — сначала coverage audit**. Третью идею «про часы/режимы» нельзя честно продавать как новую на прочитанном журнале.

### Cited Findings

- Partial-label выбор кандидатов уже E107(+0.35,6/6 хуже). E97/E111 прямой mixture-SSE имеет отрицательные реализации;№69 правильно ограничивает вердикт clipped surrogate, не все возможные likelihood-модели. MDN E196/E198 прошёл компонент, E200 провалил текущуюцепочку. Простое переименование в interval likelihood не даёт нового наблюдения или идентифицируемой latent physicaltime. [Журнал№11/23/69/335/344](../../HYPOTHESES_LOG.md), [E200](../../analysis/e200_mdn_results_20261007.md).
- E88/reception aggregates, готовые firstmove/standwalk offsets, общие соседские residual states, scalarturnaround и калибровка поARR уже проверены. ASUS№362: готовые proxy хужеbody+20.52/+23.60; оракул−0.689/−0.862 ограничивал только эти proxy, не все модели сырого трека. [Журнал](../../HYPOTHESES_LOG.md), [ASUS](../../analysis/adsb_stage_audit_20261007/RESULTS.md), [FableC](../../analysis/review_fable_2026-10-07/fable_C_error_budget.md).
- Текущий exact bestv12075 воспроизведён zero-fit. Полного nested full-tail OOF нет; sampler/ARR обязаны пройти E203preE81 comparator, а не старый residual map. E201 отменён; новое исследование не обосновывает его запуск. [Актуальный бриф](../../REVIEW_BRIEF.md), [валидация](../../docs/validation.md).

### Inferences

- Ранжирование здесь по доступности новой информации и силе различающего эксперимента. Измеренного источника многосекундного выигрыша у двух новых механизмов нет. СвободныеGPU/CPU не заменяют preregistration.

### Gaps

- Новые таргетные метрики не вычислялись, fits/production/submissions не изменялись, commits не делались. Созданы только эти notes, два metadata helpers, ихJSON иevidence receipt.
