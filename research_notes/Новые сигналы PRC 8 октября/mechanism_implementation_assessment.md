# Минимальная реализация sampler screen

**Реальный fill-only vs saved-ordered screen сегодня возможен на существующей инфраструктуре.** Нужен адаптер подготовки входов и отдельная immutable specification; новый trainer, библиотеки или получение данных не нужны. Изменять frozen E194/E197/E202/E204 files не требуется. Ниже code inspection, не измеренный runtime нового sampler.

## Что переиспользуется

- `data/e194_inputs_20261007_s0`/`s1`: daily label-free surface+departure payloads, уже на171/175. На Mac фактический disk usage2.7G+2.7G. Границы, identity, source rows, raw12channels из `scripts/e194_trajectory_sequence.py`.
- `data/e194_sequence_20261007_s0`/`s1`: original selected indices/ranges/IDs и receipts (1.3G+1.3G наMac), нужны для old⊆new и unchanged-small-stream contracts. Только2025/365days нужны для screen;2026ranking extraction не нужен до отдельногорешения.
- E197 query bundle (156M наMac), frozen E190 tabular tensors, train/pred indices, train-only numerical bins, meta/masks, saved O+seeds. Для1/4 уже всё готово; остальныеfolds живут вE202.
- Чистые функции `S.norm_stats`, `S.normalization`, `S.normalize`, `S.gpu_tables`, `S.network`, frozen fit helper и metrics. E204 overlay показывает, как переиспользовать fit/gates/receipts не трогая оригиналы.

## Что нельзя переиспользовать как готовый новый вход

Existing E197 prepared values/known/present содержат толькоoldselection. В них нет новых raw points. Нужно заново собрать именно три trajectory NPY и train-only normalization для каждого fold/arm; query tables не пересчитываются. Именаmanifest/protocol/sourceSHA будут новыми: нельзя выдавать новый sampler за прежний E197preparedreceipt.

Одна fulltrain tensor table имеет2085047×2×64tokens×(12×(2float16+1uint8)+1present)≈9.875GB=9.20GiB, плюс небольшиеids/support/receipts. Дляfill-only двухfolds≈18.4GiB; дляF/T screen четырёхtables≈36.8GiB наHome; выбранныйF-only sixpair имеет6tables≈55.2GiB наHome. Это не новыеMacкопии. Serialprep/fit может ограничить live table одной9.2GiB, но удалятьстарыеартефакты только отдельнойоперационнойполитикой; длявыбранногоsixpairбезопаснее заранее иметь≈65GiBHomeсвободно, включаяfreshrawNPZ+models.

## Наименьший адаптер

1. Новый purpose-specific script сpure sampler F/T и extraction wrapper. Импортировать frozenE194 в изолированныйprocess/module; заменить sampler только локально. `encode_day` уже даёт все12каналов/источники/маски. Own wrapper пишет fresh receipts ссвоимprotocol, не используетE194receipt какдоказательство новогоsampler.
2. В этомжеscript overlayprep проверяетoldsubset/actualrawvalues/ID/365days; используетE197normalization/mmaplayout, но ownmanifest/paths. Сохранённыйoriginalexpectedoutput используется только дляcontracts, не какtarget.
3. Отдельный небольшойrunner пообразцуE204: original `S.network(...,arm='ordered')`, новые tables, теже fit args, savedO, одинF иF−Ocomparison повсем6парам. ExistingHome systemd/cgrouplaunch применяется безизменениясреды. Eventarm сейчаснекодировать.
4. Targetedtestfile: deterministicselection/full64/oldsubset/duplicateandmissing/chronology/no-future, exact≤64, crossfitnormalization, armcoverage/gatecomparisons. Синтетический smoke доtargetfit. Протокол+журнал+все3productionfiles+tests однимreviewedcommit передCV.

Итого ориентировочно3новыхsource/testfiles (можно2приодномscript+tests) +1protocol +1journalentry, новыеmanifest/resultsdirectories. Это оценка минимальных responsibilities, не обещаниестроккода. Существующиеобщиеscriptfiles остаютсяfrozen.

## Подготовка, объём и время

Существующее полноеE194извлечение427days заняло138/166с надвухноутбуках; E197screen normalization/tensorprep ранее≈57/66с/fold. Это measured historical baselines, а не таймингновогоselection.

Рассмотренный и отвергнутый доtarget farthest-time insertion делалдо~20итераций/stream и могувеличитьPythonoverhead. Дляминимального теста root выбрал **доtarget/preregistration** фиксированныйO(n) unselected-raw-index quantile filler:

```
old = original_sampler(t)
remaining = setdiff1d(arange(len(t)), old)
need = 64 - len(old)
extra = remaining[rint(linspace(0, len(remaining)-1, need)).astype(int)]
new = sort(concatenate([old, extra]))
```

`need≤len(remaining)`, поэтому roundedindices distinct; endpoint/tie behaviour покрытьtest. T сначаладобавляетcrossingbrackets пофиксированномуpriority, затемтотжеквантильныйfiller. Это не модельноеизменение илиновыйподбор; правило надоокончательно выбратьдоtargetfit. Основнойdraftprotocol уже переведён наэтотO(n)вариант; выбранныйinitialrunFvsOвсе6пар(18fits), T остаётсяотдельнымfollow-up.

ПриO(n) filler рабочаяоценка label-freeextract+verify≈5–15мин на171/175; затемпо3foldsнакаждом, новыйnormalization/tensorprep≈5–10мин плюсLANtransfer. ИзменённыеcompressedrawsequenceNPZ ориентировочно3–5GB/F (неизмерено; original2.6GB включает427days). Можно писатьdaychunks/prepare напрямую, но это больше новогоstreamingкода; reuseNPZ дешевле инженерно. Preparedtables передавать напрямуюHome, нечерезMac. 55.2GiBпри40MB/sзанялибы≈25мин передачи; этоplanningestimate изизвестнойLANскорости, новыйpeertransferнеизмерен. Передачи/prepпоследующихfoldsможноперекрытьсGPUfitготовогоfold.

**Свежая read-only Home проверка08.10:** `/dev/nvme0n1p2`915Gtotal/206Gused/**663Gavailable**. НаHomeготовыoriginalpreparedE197=19G/E202=37G, обеrawE194sequenceпапкипо1.3G. ПоэтомуfreshF6tables≈55.2GiB+raw/receipts/models помещаютсяспбольшимзапасом; очисткастарыхtableне нужна. Команды:`ssh home-pc-lan 'df -h ~/prc-2026'` и`du -sh`толькоэтихчетырёхпапок;targetfit/новыесервисы незапускались.

Training: выбранныйF-only18fits≈72–75мин поE204baseline; прежниеbudgetoptionsF6fits≈24мин/F+T12fits≈48миннеявляютсявыбраннымprotocol. Fullworkflowplanningbudget~2ч послеreview/registration, hard6h. Подготовкаinputcode+tests ещёне выполнена. Поэтому можно запуститьscreen сегодня, но не называтьочередьработающей доunit/PID/progress.

Sources: [E194 execution](../../analysis/e194_trajectory_execution_20261007.md), [E197 execution](../../analysis/e197_trajectory_execution_20261007.md), [E204 overlay](../../scripts/night_gpu_20261007.py), [E197 prep](../../scripts/e197_trajectory_encoder.py:189), [draft scientific protocol](mechanism_sampler_protocol.md).
