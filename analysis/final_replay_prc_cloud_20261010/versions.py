import sys, importlib, platform
print('python', sys.version)
print('platform', platform.platform(), platform.machine())
for m in ['torch','numpy','polars','lightgbm','tabpfn','pandas','sklearn','scipy','pyarrow','catboost','einops','huggingface_hub','threadpoolctl']:
    try:
        mod = importlib.import_module(m); print(m, getattr(mod,'__version__','?'))
    except Exception as e:
        print(m, 'MISSING', type(e).__name__, e)
