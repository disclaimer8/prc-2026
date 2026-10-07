# E201 Home — один технический повтор после OOM

Первый Home unit `prc-e201-donors-home-20261007.service`, invocation
`ebf174d4f933451987e314d91be999bd`, начал12:50:07Z и убит systemd-oomd
12:57:48Z после7м42с/peak16GiB. Полного donor-fold и outer-оценки не получено.
Partial `.v60_fold3.partial-zr74rplj` и журнал сохраняются. Mac/Hetz идут отдельно.

Фиксированный technical repeat использует тот же immutable scientific worker,
registration `d664e153b3f0732b4047011cc062f58f42581a6c` и input manifest
`f8696b7326f196ba72f51e44be5a6b658c8f063fcac09e1bd54274c80218a2aa`.
Меняются только Homethreads12→6 и process-local glibc arena/trim limits
`MALLOC_ARENA_MAX=2`, `MALLOC_TRIM_THRESHOLD_=131072`. Общий лимит остаётся
16GiB,swap0,nice10;GPU/E202 получает прежний приоритет. Среды не изменяются.
Worker остаётся byte-identical. Этот повтор не выбирает по ошибкам ни признаки,
ни сиды, ни rounds, ни модель. Cross-platform/thread bitmatch не обещан;
исходная декларация recipe-faithful сохраняется.

`scripts/e201_home_memory_recovery_20261007.py` требует старый unit
failed/MainPID0/Resultoom-kill и точную invocation; затем пишет в новый
`data/e201_home_20261007_memoryfix`. Все folds3→4 считаются с начала,
старые partial-модели не подмешиваются. Те же audit/P/seed-арифметика/smallpacket
после полногоfold. Это один явный технический повтор после регистрации;
после очередной ошибки автоматического перезапуска или снятия лимита нет.

На момент написания код готов и py_compile PASS, **повтор ещё не запущен**.
Коллектор должен явно брать Homeпакеты из memoryfixoutput с новым operational
receipt; старый incompleteoutput нельзя объявлять завершённым.
