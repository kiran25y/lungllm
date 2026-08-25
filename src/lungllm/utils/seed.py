# REPO PATH: src/lungllm/utils/seed.py
def set_seed(seed=1337):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    try:
        import torch; torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except Exception: pass
