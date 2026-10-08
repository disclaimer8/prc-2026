# Best v12075: минимальный переносимый комплект, подготовка 8 октября

Packaging ещё не закрыт обновлёнными Claude документами. `docs/reproduction.md` корректно описывает exact replay и ограничения runtime, но обещание «complete archive available on request» пока опирается на локальные разрозненные файлы: проверенного transport archive и fresh materialized source/data checkout receipt нет. Никаких архивов, копирования данных или повторного replay в этой проверке не выполнялось.

## Проверенная существующая основа

- Entry point: `scripts/reproduce_v12075_v2.py check`, затем `scripts/reproduce_v12075_v2_launch_20261007.py --root FRESH_ROOT --registration 70aff7199731c381945c45754ee70c7e3a964aa5 --launch`. Старые runner/source/cache guards сохраняются.
- Exact proof07.10 22:33Z:344841 rows,42 protected donors,0fit, SHA `90d3cd0b5a68cc8008f2142f1801857752574b525b2ff046ad6bf4089a70aee2`;101.19с,7.90GiB peak. Это existing qualified workspace proof, не доказательство нового standalone package.
- `analysis/reproduction_bundle_20261007/dependency_manifest.json`:3711prediction/provenance paths +7runner prerequisites +3comparison-only paths =3721files, **5,232,081,827bytes** до Git/runtime/license/package tools.
- Сегодня повторно проверены existence/regular files/no links/relative containment/size всех3721paths:0issues. SHA всех5.2GB повторно не читались; старый replay checking и будущая упаковка имеют отдельные full-SHA проверки. Текущий registered replay guard по7sources/spec/tests PASS.
- Required historical commits70aff719… и d45cb738… существуют и являются предками inspectedHEAD507ebeaa…; repository non-shallow. `closure.json` задаёт9required commit:path proofs. `.git` занимает43MB, но копировать его нельзя: pack содержит только точные нужные Git objects.
- Qualified Mac runtime проверен live: Python `3.12.14 (main, Sep  1 2026, 14:09:38) [Clang 22.1.3 ]`, NumPy2.5.3,Polars1.44.2,LightGBM4.7.0,macOS27.0.1arm64. Existing `.venv` не изменяется.

## Наименьший implementation scope

Новый `scripts/v12075_portable_bundle.py`, `tests/test_v12075_portable_bundle.py`, protocol/spec/receipts только в этой analysis directory. Никакого изменения frozen runner, текущих docs, existing outputs или environments. Никакого нового remote scheduler: Home действия выполняются direct SSH/SCP/rsync после preregistration.

1. Метаданные: seal explicit file allowlist из existing dependency_manifest; три исходные группы сохраняются, LICENSE и package source/tests/protocol добавляются явно. Ни root/home globs, ни credentials, ни `.git/config`, ни произвольная история не входят.
2. Exact Git proof closure: экспортировать только2original commit objects, деревья вдоль9required paths и9required blobs. В isolated `git init` импортировать raw objects через `git hash-object -w -t TYPE --stdin`, требовать прежние OIDs, затем exact `git show COMMIT:PATH` bytes/SHA. Unneeded blobs/parents не импортируются. Это partial historical object proof, **не полный clone истории**; он не должен объявляться прошедшим full `git fsck`.
3. Home fresh staging: повторно сверить SHA available sources и скопировать их; недостающие файлы перенести по exact list с Mac. Старый Home audit:1311matching2.503GB,2400missing2.685GB плюс runner/comparison≈44.5MB; эту доступность перепроверить при staging. Не менять data в текущем рабочем Home root.
4. На Home собрать uncompressed tar с manifest первым, exact regular members, normalized metadata и SHA каждого member. Stream verifier отвергает links/hardlinks/devices/traversal/duplicates/extra/missing/wrong-size/wrong-SHA; generated archive не считается готовым до полного повторного чтения.
5. На Mac потоково получить archive в **новый** materialized checkout без локальной tar-копии. Проверить archive manifest against preregistered local manifest, every member SHA, source Git proof и old registration guards. Никаких source/data symlinks наружу. Existing qualified interpreter передаётся абсолютным путём; это не новая clean-machine installation.
6. Только после package gate выполнить existing zero-fit launcher в новом root и независимо проверить both completion receipts, их hash linkage,0fits,344841rows,42protected и конечный SHA90d3cd0…. Reference v12075 остаётся comparison-only: original runner читает его после seal generated final; premix references проверяются после generated premix. Архив не превращает reference в prediction input.

## Ресурсы и границы

Home staging≈5.3GB+tar≈5.3GB+малые proof/receipts, резервировать≥12GiB. Builder/hash tasks:1CPU,1GiB RAM,swap0,nice10,30мин, one-shot, own systemd unit; GPU/E207 не трогать. Stream buffers8MiB; package не загружает Parquet/NPZ arrays в память. Архив снаружи не публикуется: документация отмечает challenge/OPDI redistribution restrictions.

Mac live `df`:≈15GiB свободно. Один extracted payload≈4.87GiB+source/proof+existing replay output baseline135MB оставляет запас над обязательными4GiB. Нельзя хранить ещё одну5.2GB tar-копию или устанавливать новую environment1.5GB. Перед действием проверять required bytes+4GiB floor заново; во время extraction/replay не опускаться ниже4GiB для нового package wrapper (старый launcher имеет2.5GiB аварийный floor, но package policy строже).

Existing replay supervisor:6threads, monitored16GiB combined RSS,4h, start при≥16GiB reclaimable RAM и≥10% memory free; measured replay101.19с не обещает такой же timing нового checkout. Home/Linux replay не заявляется: там Python3.12.15 и нет Mac supervisor tools, cached full sys.version guard отклонит его. Home служит только сборке/проверке bytes.

## Что является завершением

Private transport artifact + explicit manifest/license/source proof + whole-archive checksum receipt + clean materialized checkout receipt + independent new-root zero-fit exact replay receipt. Компоненты не переобучаются, новая модель не выбирается. Название доказательства: **fresh materialized source/data checkout on the existing qualified external Mac runtime**. Полный public-data raw retrain, cross-platform equivalence, whole Git-history clone и новая clean-machine installation остаются отдельными задачами.

Root владеет журналом/prereg commit/запусками/обновлением текущих docs. До этого разрешены только new code/protocol/spec metadata и synthetic tests; archive/staging/extraction/replay запрещены. Публикация и отправка организаторам остаются отдельными действиями.
