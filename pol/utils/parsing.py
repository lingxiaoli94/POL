import torch

def list_to_tensor(L, dtype=torch.float32):
    if isinstance(L, list):
        return torch.tensor(L, dtype=dtype)
    return L
