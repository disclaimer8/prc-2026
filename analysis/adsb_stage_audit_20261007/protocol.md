# Аудит ADS-B-стадии: наблюдаемое первое движение и оракулы

Protocol ID: `ADSBAUDIT-E195joint-firstmove-crosshalf-lag-oracle-airportcoverage-v1`.
Worker: `scripts/adsb_stage_audit_20261007.py`; sealed inventory:
`analysis/adsb_stage_audit_20261007/input_receipt.json`.
Code + protocol + receipt + запись журнала регистрируются до новых сравнений.
До регистрации разрешены schema/source inspection, hashing и synthetic tests.

## Вопрос и граница

Предложение FableA/H2a и FableC: на актуальной цепочке измерить остаток по
аэропорту и реальному классу покрытия, сравнить с простой физической оценкой.
E48 измерял покрытие на старой E28/E30, поэтому новый аудит обновляет контроль.
E36/E70/E74/E141 уже используют firstmovement,geometry,visit,standwalk; это
**не регистрация нового датчика/модели** и не повтор закрытого residualfit.
E39/E134prefix,E179softgate и E183neighbors не переоткрываются этим диагнозом.

Контроль — exact `joint` из E195 `data/e195_pre_e81_20261007_v2/predictions.parquet`,
SHA `271b01811a624dc1eb01aa8c98d249d97a7eb2d3814062860beb54b3e3787ebb`.
Все2 085 047ID сохраняются; сравнения только на2 084 921included,126TabPFN
исключаются точно какE195, feasibilityexclusions0. Проверяются уникальность,
ID/y/month/airport/NM и input/outputSHA. Это afterE152/preE81 jointOOF,
**не полныйv12074**, без E81/финальныхLOBT/TabPFNprotection/halfLSZH.
UpstreamOOF исторический, не полностьюnested; никакого новогоMLобучения.

Наблюдаемые признаки — sealed E195 `natural_features.parquet`,SHA
`a82c14fc733b6f7355b9c7cca65a8cf223d13d2a05d7dba5adeb73420e952a16`, stage spec
`5c4a0281a6669d856331317128917097b2a87df55a5c44f3cb0f0d26602441b2`.
Исходные методы и неизменные parent receipts закреплены, новая ADS-Bэкстракция
не запускается. Ranking используется только для безметочной таблицы покрытия
по тем же классам:62дня исходного offblock2 и62дня1ktstandwalk,
projected columns FIN1b. Оба источника используют1kt: `data/adsblol_kt1` и
`data/adsblol_sw_kt1`, как frozenE195 builder (`scripts/e195_current_chain.py:121`).
Его sourceSHA и sensorpaths в parentmanifest повторно сверяются приseal. Rankingtarget не читается, прогноз ranking не создаётся.

## Фиксированные моменты и классы

Primary: `mvt_first_move = MVT − t_first_move`, исходный1ktдетектор:
первое движение после начального неподвижного отрезка.
Secondary, всегда показывается отдельно: `sw_mvt_first_move`, принятый1kt
standwalk. Не выбирать лучший вариант после результата и не смешивать их.
`mvt_move` обозначает MVT минус конец **последней** достаточно длительной
остановки, а не первое движение; только сохраняется для семантической проверки.

Coverage−1: missingfeature/day;0: нет однозначного сопоставленного трека;
1: нет принятого наземного сегмента;2: сегмент есть, нет квалифицирующей остановки;
3: до последнего движения найден неподвижный отрезок≥90с. Класс3 сам по себе
не доказывает наблюдение у стоянки или правильность recordedBLOCK.
Оба поля оцениваются по исходномуcoverage; secondaryswcoverage не подменяет
основную группировку. Eligibility зависит только от исходногоcoverage≥2,
finite duration в[0,43200]с и исходнойE195evaluationmask. Неподдерживаемые
строки сохраняют exactjointfallback, не получают выдуманный firstmovement.

## Наблюдаемые оценки без MLfit

1. Zero lag: proxy=movementduration.
2. Cross-half median lag: строго Jan–Jun→Jul–Dec и обратное направление.
   На traininghalf eligibleincluded строках lag=movementduration−y.
   Median по airport×coverage; минимум100trainingobservations, иначе
   coverage-only, затем globaltrainingmedian. Никаких stand/runway bins,
   оптимизации порогов, регрессионного обучения или выбора по heldouterrors.
   Proxy=max(0,movementduration−lag); отрицательный lag добавляет время
   междуrecordedBLOCK и наблюдённым движением. Traininglabels могут быть
   регулярными/плановыми: **нет фильтра heldoutregime/y для предсказания**.
   Для каждойlookupcell сохраняются source/fallback,n,lag,exacttrainingIDhash.

Полугодовое разбиение относится только к новым lagstatistics. Оно не превращает
исходный E195jointOOF в независимое nestedhalfCV. Heldouty используется только
после созданияobservableprediction для сравнения и отдельно названныхоракулов.

## Оракулы — отдельно, никогда не deploymentgain

Для каждого заранее заданного evaluationcell/population:

- Label-aware optimal constant lag = mean(movementduration−y) **на этих же
  оцениваемых строках**. Сообщается RMSE un-clippedunit-slopeconstant-offset
  family с exactjointfallback внеeligible. Это оптимум только указанного
  семейства и оценочной ячейки; не неснижаемый физический шум и не потолок
  любой ADS-Bмодели. Вbodyoracle самy-фильтр также недоступен припрогнозе.
- Perfect per-row choice междуjoint и фиксированнымcross-half/zero proxy
  выбирает меньшую squared error с помощьюистины. Это недостижимый предел
  идеальногоselector именно этих двухготовыхпрогнозов, неготовыйалгоритм.

Оракулы не входят в lagtable,observableprediction,diagnosticcontinuation flag
или rankingprojection. Никакая оценка по y не называется наблюдаемым выигрышем.

## Таблицы и ограниченное решение

На included/body(y≤7200)/decided(E195definition) публикуются RMSE/bias/SSEdelta,
matchedeligiblecount, symmetrictrim3/20SSE для группы, обаоракула.
Группировки фиксированы: общийпул; airport×coverage; airport×coverage×NM;
12months;2halves. Мелкиеtrimgroups получаютnull приn≤2k.
Нет airportwiseбленда или выборочного устранения плохихячеек.

Первичный вопрос для **отдельного будущего опыта**, не внедрения:
уprimarycross-halfmedian bodySSE<0 в обоихнаправлениях и pooleddecidedSSE<0.
Secondaryотчёт имеет ту же описательнуюпроверку, но не заменяетprimary.
Любойисход завершает этотaudit, не запускаетfit/ranking/upload автоматически.
Prior ΔbodyRMSEprimary0с,экспертныйнекалиброванныйкоридор[−1,+1]с;
потенциалFable2–5с не считается измеренным или нашим обещанием.

Rankingтаблица показывает rows/eligible поairport×coverage×NM. Дополнительно
фиксированное reweighting2025bodyMSEdelta долямиrankingячеек, с явным количеством
unsupportedcells/rows. Это **описательная source-yearперевзвешенная энергия
ошибки**, не прогноз RMSEдоски и не доказательство переносаприreceiverdrift.
В частности, LIRFJuly/2026 иLTFM нельзя заполнять чужимтрекам.

## Выполнение и ресурсы

Mac только `seal` и после регистрации `pack`: projectedtraining/rankingcolumns,
безlag/errorcomparisons. Portablebundle в этойпапке/`portable_inputs` содержит
точные parquet/source/protocol/receipt и manifestSHA. Многогигабайтные query
tensors, models и весьFIN1b наASUS не копируются.

ASUS Ally: `C:/Users/admin/prc-2026/.venv-prc/Scripts/python.exe`, существующая
среда; CPUaffinity[0,2],BelowNormalpriority,Polars2/BLAS1/OMP2.
KernelWindowsJobObject hardprocesscommitted-memorylimit6GiB; helperскопирован
из уже используемого `cat_ally_windows.windows_limit`, **никакойCatqualification
илиCatfit**. ПередработойACplugged и≥3GiBavailable, измеренныйRSSguard2GiB.
Все source/dataSHA проверяются перед и после; noenvupdates/noGPU.
Macrun не нужен: возвращаются небольшиеtables/rows/report срезультатом.

Run требует originalregistration и exacttransportmanifestSHA, freshoutput,
пишет invocationPID/runtime/resources до расчётов; final atomicrename только
после всех проверок. Existingoutputнеперезаписывается, noautomaticretry.
Synthetictest проверяет, что любые измененияheldoutlabels не меняют observable
lag, а также eligibilitybounds, sign иperfectselectorarithmetic.
