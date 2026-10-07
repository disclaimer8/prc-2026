# E201 — восстановление двух доноров E81 с новым cross-fit

Протокол `E201-v44-v60-recipe-donors-inner-rounds-six-pair-exclude126-v1`.
Worker `scripts/e201_donor_oof.py`. Регистрация до любого нового model-fit.
Это **воспроизведение рецептов v44/v60 на held-out месяцах**, не восстановление
байт-в-байт старых артефактов и не новый предиктор для подачи на leaderboard.

## Зачем нужен расчёт и что найдено

E195 восстановил текущие четыре arms доE81. Сам E81 смешивает main с двумя
донорами: main0.8516, v44 0.1214, v60 0.0270. В старых OOF E38/E46 и
`cv_preds_nested_ap` есть лишь IDs/pred/reg/y; вероятности режимов не сохранены.
Для ADS-B stages нужен настоящий p_nonreg. Подставлять его из `nomonth` другой
модели нельзя: существующий `cv_preds_e46_proba_from_nomonth` — не точныйдонор.
Ранжирование v44/v60 сохранилось, но не является OOF2025.

Рецепт найден в `src/run_final_any_period.sh:431–452`, `run_final_e38.sh`,
`after_v42_stage.sh`, `run_final_e52.sh`, `submit_per_airport.py`, `blend.py`.
Исторические snapshots берутся из git, не из меняющихся currentsrc:

- v44: `da209d5ae10edfef6958a16fc1c593a12b936efa` (15/16сентября),
  train/features/target_encoding, E14, perairportE42.
- v60: `1c35ed46d70eac38152ea8f3a34e24d4f4b6f258` (16сентября),
  train/features/target_encoding, E14, globalE36.

Существующие `features_e28.parquet` и `features_e32.parquet` содержат одинаковые
2429888ID/month/y в том же порядке:2085047train,344841rank. Файлы645/688MB.
364дня каждого3kt offblock2/3/5 присутствуют; в каждом наборе2080332uniqueIDs,
без дублей. 4715рейсов31декабря не имеютсенсоров и получают stagefallback.
Все шесть `cv_preds_nomonth` имеют настоящую6-column proba и полный2085047ID.
Новая manifest фиксирует SHA всех фактически использованныхinputs и git-blobs.

## Границы воспроизводимости

Старый E38 выбирал early-stopping на оцениваемых outerмесяцах. Повторять это
нельзя. Сохранённые productionrounds тоже выбирались поJan/Jul2025: они уже
смотрели на outer1. Поэтому они служат только историческим свидетельством,
а новый расчёт выбирает rounds во внутреннем разрезе.

Runtime/полнота текущих3kt сенсорных архивов могли измениться ссентября.
Новыйдонор следует архитектуре, признакам, стадии и арифметике исходного рецепта,
но не утверждает численного совпадения со старыми файлами v44/v60 на ranking.
Старые rankingфайлы,42TabPFN replacements и productionсборка не меняются.

Stages E36/E42 по оригинальному рецепту учатся на историческом `nomonth` OOF.
Эти upstreamпрогнозы не пересчитываются во вложенном разрезе каждого outerfold:
их модели могли видеть labels текущего outer через другиеfolds. Это inherited
retrospective limitation, **не fullynested whole-chain CV**. Новые donorbase/
proxy/encodings/innerrounds действительно исключают свои outerlabels.

## Точные splits и округления

Все шесть pairs(a,a+6),a=1..6. Внешнее training — остальные10месяцев.
Внутренняяvalidation — (a%12+1,(a+6)%12+1) внутриэтих10, innertraining —8месяцев.
Ни одна outerметка не участвует в roundselection. Все targetencoding/proxy
калибровки внутри T.fit_predict строятся только по соответствующему fit.
Классификатор/regressor inner: максимальные5000rounds, early_stopping200,
как historical train.py. Внешнийrefit int(best*1.1), минимум1, без раннейостановки.

Перед внешним encoding/inference held-out Y заменяется null; labels сохраняются
отдельно для receipt/дальнейшей проверки. Встроенный guard требует, чтобы
исторический add_encodings не менял порядок ID при текущемPolars.
ВсеfitIDs/predIDs и месяцы сверяются; innerIDdigests сохраняются.

После смеси применяется исходное LOBTmask/renormalization/clip≥0 и округление
Polars round(0) как в production. Затем E14 updates отдельно округляются,
потом stage correction и ещё одно productionround(0). Во всех P/values/reg
сохраняется float64 до соответствующего явного округления.

## Донор v44

E28features, DROP_PREFIX=`day_,tq_,wxr_,month`. Отдельныеclassifier+regressor
для каждого из10airport. PROXY_SCALE выключен, используются medianoffsets.
Исходныеparams: reg511leaves/lr.03/minleaf50, clf127leaves/lr.05/minleaf100,
bagging.8/freq1/featurefraction.8/L2=10; seed42, deterministic,row-wise.
Каждый airport+outerfold:2innerfit+2outerfit. Итого240LGBfit по6folds.
Нет подмены аэропорта глобальноймоделью и нет пропуска слабого аэропорта.
Реальные proba/values/reg сохраняются для каждой outerстроки.

Далее original E14 на LIRF/noNM,0<D<14400, historyn≥3: logit C1/maxiter2000,
out-of-month history внутри10trainingmonths; validationhistory из этих10.
Prediction=p_sched*D+(1−p_sched)*donorreg. СтрокиD≥14400 не трогаются.

ПослеE14 — originalperairportE42: trainнаhistoricalnomonthOOF других10месяцев;
отдельныйairportmodel только если trainingcoverage≥2 содержит>2000строк.
В modeltrain |y−base|≤3600, targetclip±1800. OriginalFEATS=E30+standgeometry,
63leaves/minleaf200/lr.03/800trees/seed42. Применениетолькоcoverage≥2;
без подходящей модели/покрытия сохраняетсяprestagebase. Наapplication
base и p_nonreg берутся от НОВОГОv44донорa, base-relativefeatures пересчитываются.

## Донор v60

E32features, тотжеDROP_PREFIX, PROXY_SCALE=1; airport/source коэффициенты
сridge1000 заново оцениваются только на donortrainingordinaryrows.
Одно globalinnerCLF+REG, затем fullouter10month refit classifier.
Ordinaryregressor:5LightGBM seeds1000..1004, baggingseeds2000..2004,
featurefractionseeds3000..3004 с выбраннымinnerregrounds.
Плюс **один CatBoost seed0**, iterations5600,lr.08,depth8,RMSE, CPU.
Это явно задано историческим `catboost_reg.py`, не предположение о defaultseed.
Target = clip(clip(y,0,7200)−proxy,−7200,+7200), как в обоих источниках.
Catreg0.65+LGBmean0.35. После замены ordinarycandidate заново mask/normalize
реальныйclassifierP и mixturevalues, затем productionrounding.
Categoricaldictionary classifier строится на всехfitrows, а LGBseedregressors —
ОТДЕЛЬНО только наordinaryfitrows, как historical seed_avg.py; nonregular-only
categories вregressor prediction остаютсяunknown. Cat используетstrings/NA.

Одноouterfold:2innerLGB+1outerCLF+5outerLGB+1Cat=9fits;
всего54fits по6folds, в том числе6Cat по5600. Не обучать лишний singleLGB,
который historicalsubmit потом всё равно заменял five-seedmean.
CatPool categoricalstrings/missingNA как production. Stored per-seed predictions
и модели обязательны, Cat runtime фиксируется вreceipt.
Затем тотжеE14 routing и globalE36 800trees, обученныйна historicalnomonthOOF
других10месяцев; applicationbase/P от новогоv60, ownrelativefeatures пересчитаны.

## TabPFN и оценка

Все ID сохраняются. Общаямаска126TabPFNстрок берётся из pinnedE195result
SHA `94ba00d1522b7d598bc9d3b42d3361ad68af9dd41ca19b120c5c07f023e05f9e`,
проверяется predictionSHA из этогоresult; feasibilityexclusions должныравняться0.
На этих126строках donorбезTabPFN не выдаётся за точную цепочку: evaluation_mask=false.
Не использовать старый TabPFNcache сglobalmedian как productionзамену.

Это infrastructure reconstruction, не гипотеза приростаRMSE. Prior ожидаемого
изменения body для последующих сравненийarms0с, экспертныйкоридор[−1,+1]с,
некалиброванный/непорог; прогнозкачества новыхdonorsневыдумывается.
Гейт этойфазы — полные12donorfolds,2085047uniqueIDs накаждомдоноре,
настоящая6classP ссуммой1, точныеmodelinventories/seeds/roundpolicy, отсутствие
outerfitlabels, SHA/порядок/конечность/общая126маска. Не выбирать donors/rounds
по их наружным ошибкам. Завершить все фиксированныеfolds независимоотRMSE.
Последующий postE81 comparison долженявно указывать новый scoped donor lineage;
при missingfold/P/артефакте завершение не публикуется. Submissionнеразрешена.

## Вычисления и очередь

Фиксированно бездублирования:171→v44folds1/2/3;175→v44folds4/5/6;
Mac→v60folds1/2;Home→v60folds3/4;Hetz→v60folds5/6.
Накаждомузле один donorfoldworker одновременно; withinfold v44airportmodels
последовательно, потомE14/stage, поэтому полезное продолжение не ждёт другихузлов.
После всех12folds коллектор создаёт общийdonors.parquet иverifiedcomplete.

171/1756CPU/10GiB/swap0/nice10+sleepinhibitor; Home12CPU/16GiB/swap0/nice10;
Hetz6physicalcores0,2,4,6,8,10/16GiB/nice10,OMP_THREAD_LIMIT6/PASSIVE/spin0;
Mac8threads подcaffeinate, отдельныйпроцесс безTorch, контрольRAM/диска.
Mac/Hetz/171 используют исходные `.venv`; 175 — уже квалифицированную
`.venv-cat-parity`, Home — существующую `.venv-cat-oof` (в обычной `.venv`
CatBoost отсутствует). Среды не менять.
Реальныеruntimeversions сохраняются, crossplatform-bitmatch не заявляется.
E197/E200GPU не затрагиваются. E200candidateassembly/submission имеетприоритет.

Запускпосле своих E199назначений (включая условноеcontinuation или evidenceFAIL),
при activeE200CPU на конкретном узле ждать именованное завершение; не стопатьjobs.
Очередь отдельныйdurableOScontroller с boundedtime, failurelog и noautoretry;
не держится только на агентском polling. Worker output всегда fresh/atomic;
partial сохраняется послеошибки и не считаетсяготовым.

Macpack использует read-only hardlinks к1.33GBисходныхfeatures; неизменность
защищенаSHA прикаждомrun, ниworker ниcollector inputs не пишут. Не копировать
19GiB E197packs. Дневные3ktinputs сводятся в небольшой stageparquet наMac;
на узлы передаются готовыеfeaturetables+stage/history/meta+frozen.sources.
Полные модели/seedarrays остаются на исходномузле: все294modeloutputs могут
превысить свободныйдискMac. `audit` на каждом исходномузле заново проверяет
КАЖДЫЙSHA, split/round/modelinventory, настоящиеP и точнуюCat/LGBseedарифметику,
затем пишетtransfer_receipt сполнымmanifest/completeSHA/sourceSHA/host/path.
НаMac приходят complete+transferreceipt и все небольшиеJSON/parquet/npz;
collect проверяет привязкуremoteproof иSHA каждой полученнойстрокиартефактов.
ПолнуюлокальнуюналичностьмоделейMac не утверждать; исходныемодели не удалять.
Оценка времени — несколько часов с288LGB+6Cat=294basefits (и до66ADS-Bstages), не обещание
постоянной загрузкивсехCPU доночи. ФактическиеETA уточнятся поreceipts.
