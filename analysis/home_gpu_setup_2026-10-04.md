# Домашний GPU: проверка 04.10.2026

Ryzen 9 7900X3D, 24 логических CPU, 30 713 MiB RAM; Radeon RX 7900 XTX 24 GiB.
Ubuntu 26.04.1, kernel 7.0.0-38-generic. Существующий inbox `amdgpu` и `/dev/kfd` работали;
пользователь a1 входит в video/render. Системный драйвер не менялся и перезагрузка не требовалась.

По просьбе Denys LLM временно остановлен: `flight-llm-api.service` и `ollama.service`.
Остановка требовала локального sudo-пароля; пользователь выполнил команду, затем SSH подтвердил
`inactive`, `MainPID=0` у обоих. Сервисы не удалены и не отключены в автозапуске.

Создана отдельная `/home/a1/prc-2026/.venv-rocm` через существующий Python 3.12.15.
Официальная установка AMD: `pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ 'torch[device-gfx1100]==2.13.0+rocm10.0.0'`.
Torch сообщает HIP 7.15.26333, `torch.cuda.is_available()=True`, `cuda:0` RX 7900 XTX.
ROCm использует API torch.cuda; это AMD-ускорение. Второй GPU — встроенная Radeon Graphics;
для обучения он не выбран. Рабочая `.venv` текущего E185 не менялась.

Официальные источники: [AMD PyTorch](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html),
[матрица ROCm](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html),
[требования к драйверу](https://rocm.docs.amd.com/en/docs-10.0.0/install/rocm.html).
AMD документирует vendor-драйвер для Radeon; успешная проба ниже подтверждает конкретную
существующую inbox-конфигурацию, а не полное соответствие vendor-установке.

Синтетическая архитектура E186, FP32, 272 числовых и 14 категориальных входов, batch512,
16 ensemble heads, три прогревочных и десять измеренных шагов. GPU синхронизирован перед
и после каждого шага. CPU/GPU forward maxabs **3,73e-8**, градиенты **3,73e-9**, все конечны.
Медианы: GPU **0,0035983 с**, CPU4 **0,520565 с**, отношение **144,67**. CPU сравнивался
на четырёх потоках при параллельном E185; это не сравнение со всеми 24 CPU и не ускорение
полного пайплайна. Пиковый GPU allocator 281 546 752 байт. Подготовка настоящей FIN1b,
перенос minibatch и полное CV этим тестом не измерены; метки конкурса не читались.

Воспроизведение: `.venv-rocm/bin/python analysis/home_gpu_probe_2026-10-04.py`.
Точные измерения и SHA кода: `analysis/home_gpu_probe_2026-10-04.json`.

LightGBM GPU требует отдельной сборки и измерения; установленный CPU LightGBM автоматически
на GPU не перешёл. Текущий E185 продолжается с исходным числом потоков и параметрами.
