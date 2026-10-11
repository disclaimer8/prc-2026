import torch, numpy as np, json
cfg = np.show_config(mode='dicts')
print('numpy blas', json.dumps(cfg['Build Dependencies'].get('blas')), 'lapack', json.dumps(cfg['Build Dependencies'].get('lapack')))
print('numpy simd', json.dumps(cfg.get('SIMD Extensions')))
s = torch.__config__.show()
print('torch', ' | '.join(l.strip() for l in s.splitlines() if any(k in l for k in ('BLAS', 'LAPACK', 'MKLDNN', 'oneDNN', 'ACL', 'OpenMP', 'CPU capability', 'NNPACK', 'XNNPACK'))))
print('torch blas/mkldnn/acl', torch.backends.mkldnn.is_available(), getattr(torch.backends, 'mkldnn').enabled, torch.get_num_threads())
from threadpoolctl import threadpool_info
print('threadpools', [(d.get('internal_api'), d.get('version'), d.get('filepath', '').split('/')[-1]) for d in threadpool_info()])
