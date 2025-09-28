import torch, random, numpy as np

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def to_device(batch, device):
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
