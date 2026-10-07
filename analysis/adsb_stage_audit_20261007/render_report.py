"""Render only the registered audit's saved, verified tables."""
import json,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
r=read(B/'results/result.json');v=read(B/'verified_results.json');c=read(B/'results/comparisons.json');s=read(B/'results/support.json');terminal=read(B/'execution/observed_terminal.json')
fields=['mvt_first_move','sw_mvt_first_move'];names={fields[0]:'Первое движение (primary)',fields[1]:'Standwalk (secondary)'}
def get(f,pop='body',grouping='all',group=None,arm='cross_half_median_lag'):
    return next(x for x in c if x['field']==f and x['population']==pop and x['grouping']==grouping and x['group']==(group or {}) and x['arm']==arm)
def count(ap,pop,f,key):return sum(x[key] for x in s if x['ADEP_mvt']==ap and x['population']==pop and x['field']==f)
def delta_oracle(x,k):return x[k]['rmse']-x['rmse_control']
lines=[
'# ADS-B stage audit: результат 07.10.2026',
'',
'**Фиксированная замена текущей стадии на MVT − first movement − cross-half lag отклонена.** Primary ухудшает body RMSE на **20,520 с**, secondary standwalk — на **23,597 с**. Оба направления полугодового переноса, pooled decided и обе усечённые суммы ошибок ухудшились. У девяти аэропортов с наблюдениями ухудшение есть и в coverage 2, и в coverage 3. LTFM не имеет допустимых наблюдений. Этот диагноз не открывает обучение, подбор весов или подачу.',
'',
'Регистрация №360: `f2fc89603be4a1521860a98b2adb641d392a3e09`. Контроль — exact E195 joint после E152 и до E81; это **не полный v12074**. Сохранены все 2 085 047 ID, оценены 2 084 921 строки, исключены исходные 126 TabPFN. Исторические upstream OOF не полностью nested. Для ранжирования прочитаны только наблюдаемые признаки и покрытие 344 841 строки, без меток или новых прогнозов.',
'',
'## Наблюдаемые оценки',
'',
'Единицы таблицы — секунды ΔRMSE относительно сохранённого joint; положительное значение хуже. Body: y≤7200; decided: исходная маска E195. На строках без допустимого наблюдения остаётся exact joint. Half 1 оценивается лагом из half 2, и наоборот. Для каждого аэропорта×coverage лаг — training-half median(duration−y), минимум 100 строк; далее coverage fallback. Heldout y не входит в этот прогноз.',
'',
'| Оценка | Eligible included | Body Δ | Decided Δ | Body H1 Δ | Body H2 Δ |',
'|---|---:|---:|---:|---:|---:|']
for f in fields:
    x=get(f);lines.append(f"| {names[f]}, cross-half | {get(f,'included')['eligible_rows']:,} | {x['delta_rmse']:+.3f} | {get(f,'decided')['delta_rmse']:+.3f} | {get(f,grouping='half',group={'half':1})['delta_rmse']:+.3f} | {get(f,grouping='half',group={'half':2})['delta_rmse']:+.3f} |")
    lines.append(f"| {names[f]}, zero lag | {get(f,'included')['eligible_rows']:,} | {get(f,arm='zero_lag')['delta_rmse']:+.3f} | {get(f,'decided',arm='zero_lag')['delta_rmse']:+.3f} | — | — |")
lines += ['',f"Контроль body RMSE **{get(fields[0])['rmse_control']:.6f} с**; primary **{get(fields[0])['rmse_proxy']:.6f}**, secondary **{get(fields[1])['rmse_proxy']:.6f}**. Body trim3/trim20 ΔSSE: primary +15,334/+13,445 млрд с², secondary +17,970/+15,961 млрд с². Удаление единичных крайних строк знак не меняет. Зарегистрированный экспертный коридор primary [−1,+1] с не выполнен; это был некалиброванный prior, не обещание результата.",
'','## Аэропорт × покрытие','',
'Детекторы в обеих популяциях согласованы с frozen E195: **1 kt** (`adsblol_kt1`, `adsblol_sw_kt1`). Primary — MVT минус первое движение после начального неподвижного отрезка. `mvt_move` относится к концу последней остановки и в эту оценку не подменялось. Coverage 3 означает найденный неподвижный отрезок ≥90 с перед последним движением; он **не доказывает наблюдение у стоянки**. Метрики ниже относятся ко всему body соответствующей ячейки, включая неизменённый fallback.',
'','Coverage 3, где гипотеза о физическом времени выглядела наиболее привлекательной:','',
'| Аэропорт | Body rows | Primary eligible | SW eligible | Joint RMSE | Primary Δ | SW Δ | Oracle scalar primary Δ | Oracle selector primary Δ |',
'|---|---:|---:|---:|---:|---:|---:|---:|---:|']
airports=sorted({x['ADEP_mvt'] for x in s})
for ap in airports:
    matches=[x for x in c if x['field']==fields[0] and x['arm']=='cross_half_median_lag' and x['population']=='body' and x['grouping']=='airport_coverage' and x['group']=={'ADEP_mvt':ap,'coverage':3}]
    if not matches:continue
    x=matches[0];z=get(fields[1],grouping='airport_coverage',group={'ADEP_mvt':ap,'coverage':3})
    lines.append(f"| {ap} | {x['rows']:,} | {x['eligible_rows']:,} | {z['eligible_rows']:,} | {x['rmse_control']:.2f} | {x['delta_rmse']:+.2f} | {z['delta_rmse']:+.2f} | {delta_oracle(x,'oracle_same_evaluation_cell_constant_lag'):+.2f} | {delta_oracle(x,'oracle_per_row_joint_or_fixed_proxy'):+.2f} |")
lines += ['','Coverage 2, без квалифицирующей финальной остановки:','','| Аэропорт | Primary eligible | SW eligible | Primary body Δ | SW body Δ |','|---|---:|---:|---:|---:|']
for ap in airports:
    matches=[x for x in c if x['field']==fields[0] and x['arm']=='cross_half_median_lag' and x['population']=='body' and x['grouping']=='airport_coverage' and x['group']=={'ADEP_mvt':ap,'coverage':2}]
    if not matches:continue
    x=matches[0];z=get(fields[1],grouping='airport_coverage',group={'ADEP_mvt':ap,'coverage':2})
    lines.append(f"| {ap} | {x['eligible_rows']:,} | {z['eligible_rows']:,} | {x['delta_rmse']:+.2f} | {z['delta_rmse']:+.2f} |")
lines += ['','## Оракулы и граница вывода','',
'Обе следующие оценки используют истину в оцениваемой ячейке и **не доступны при прогнозировании**. Они не входят ни в lookup lag, ни в решение о продолжении. Oracle scalar — один оптимальный un-clipped offset для unit-slope duration в данной ячейке; это не предел качества любой ADS-B модели и не оценка неснижаемого физического шума. Oracle selector выбирает между уже готовыми joint и фиксированным proxy по меньшей фактической ошибке.',
'','| Полный body | Joint RMSE | Oracle scalar RMSE (Δ) | Perfect selector RMSE (Δ) |','|---|---:|---:|---:|']
for f in fields:
    x=get(f);o=x['oracle_same_evaluation_cell_constant_lag'];q=x['oracle_per_row_joint_or_fixed_proxy']
    lines.append(f"| {names[f]} | {x['rmse_control']:.3f} | {o['rmse']:.3f} ({o['rmse']-x['rmse_control']:+.3f}) | {q['rmse']:.3f} ({q['rmse']-x['rmse_control']:+.3f}) |")
lines += ['','Даже при выборе строки с помощью y потолок выигрыша **именно этого готового proxy** составляет только 0,689/0,862 с на полном body. Локальные selector-oracle выигрыши в coverage 3 не являются готовым правилом маршрутизации. Наблюдаемая cross-half оценка резко хуже контроля, а scalar oracle хуже в каждой поддержанной airport×coverage ячейке. Это отвергает простую фиксированную замену стадии, но не произвольные будущие модели траекторий.','',
'## Доступность и перенос','',
'Доля допустимых firstmovement, %, среди всех included training / всех ranking; в скобках ranking count. Это доступность именно данного момента, а не любого ADS-B трека.','','| Аэропорт | Training primary % | Training SW % | Ranking primary % (n) | Ranking SW % (n) |','|---|---:|---:|---:|---:|']
support_summary=[]
for ap in airports:
    n=count(ap,'training',fields[0],'evaluation_rows');nr=count(ap,'ranking',fields[0],'rows');t=[count(ap,'training',f,'eligible_rows') for f in fields];q=[count(ap,'ranking',f,'eligible_rows') for f in fields]
    lines.append(f"| {ap} | {100*t[0]/n:.2f} | {100*t[1]/n:.2f} | {100*q[0]/nr:.2f} ({q[0]:,}) | {100*q[1]/nr:.2f} ({q[1]:,}) |")
    support_summary.append(dict(airport=ap,training_rows=n,training_primary=t[0],training_secondary=t[1],ranking_rows=nr,ranking_primary=q[0],ranking_secondary=q[1]))
tot=[sum(x['eligible_rows'] for x in s if x['field']==f and x['population']=='ranking') for f in fields]
lines += ['',f"Всего ranking eligible: primary **{tot[0]:,}/{344841:,} ({100*tot[0]/344841:.2f}%)**, secondary **{tot[1]:,} ({100*tot[1]/344841:.2f}%)**. Среди ranking без NM поддержаны только **335** строк primary и **459** строк secondary. LTFM: 47 581 ranking строка и ни одного допустимого момента. EDDF показывает сильное падение доли наблюдений, EGLL — рост; поэтому перенос долей из 2025 без проверки неверен.",
'','Зарегистрированное перевзвешивание source-year body ΔMSE по ranking airport×coverage×NM даёт **+14 428,02/+15 899,39 с²**; 344 838 строк имеют совпадающую ячейку, 3 — нет. Это описательная энергия ошибки 2025 при новых весах ячеек, **не прогноз leaderboard RMSE**. Изменение доли допустимых firstmovement внутри ячейки и receiver drift этим взвешиванием не исправлены. Полные NM/месячные таблицы сохранены в `results/comparisons.json` и `results/support.json`.',
'','## Выполнение и проверка','',
'ASUS Ally, существующий Python 3.14.4, NumPy 2.5.3, Polars 1.44.2; PowerShell 7.6.6. Scheduler `PRC-ADS-B-Stage-Audit-20261007`, launcher PID 5212, Python PID **15204**. Фактический worker: **15:30:38–15:30:47 UTC**, **9,364 с**, peak RSS **1 347 321 856 bytes (1,255 GiB)**; affinity [0,2], BelowNormal, kernel committed-memory cap 6 GiB. Старый VRAM watcher не работал и не имел триггеров; firmware не менялась.',
'','В 15:34:51 UTC подтверждены `LastTaskResult=0`, состояние Ready, **0 triggers**, отсутствующее NextRunTime и **0 процессов**. Ready здесь означает завершённую задачу без будущего запуска. Автоматического повтора нет. Моделей обучено **0**, ranking-прогнозов создано **0**, файлов на доску отправлено **0**.',
'','Независимая проверка на Mac: **PASS** всех входных/выходных SHA, точных ID/y/month/airport/NM/control/masks, 72 lookup-таблиц с training-ID hash и непересекающимися half IDs, bit-exact proxy/fallback, **1 932** сравнительных строк (включая trims и оба оракула), **348** support-строк. Это повтор арифметики готового аудита, не новый эксперимент.',
'',
'- [Протокол](protocol.md), [receipt](input_receipt.json), [проверка](verified_results.json).',
'- [Итог worker](results/result.json), [invocation](results/invocation.json), [terminal evidence](execution/observed_terminal.json).',
'- Manifest SHA: `3bc8613751d0118e3bdf5bfed7326fc168bdabeef17a05c6a50b288cdf76794c`.',
'- Result SHA: `752052de97612d4aa9f16339dbcdfc7cb74d41b0e1d91ffd96bd628788212f92`.',
'- Большие parquet и portable bundle остаются локальными артефактами; их не добавлять в git.',
'']
(B/'RESULTS.md').write_text('\n'.join(lines))
summary={'status':'negative_diagnosis_verified','registration':r['registration_commit'],'result_sha256':v['result_sha256'],'decisions':r['decisions'],'pooled':[x for x in c if x['grouping']=='all'],'body_halves':[x for x in c if x['grouping']=='half' and x['population']=='body' and x['arm']=='cross_half_median_lag'],'support_by_airport':support_summary,'projections':r['projections'],'terminal':terminal,'verification':{k:val for k,val in v.items() if k!='artifact_hashes'}}
(B/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('Rendered RESULTS.md and summary.json; no new comparative calculation')
