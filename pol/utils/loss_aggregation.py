import torch


def aggregate(L, dim, desc):
    mode = desc['mode']
    if mode == 'mean':
        return L.mean(dim)
    elif mode == 'max':
        return L.max(dim)[0]
    elif mode == 'lse':
        assert('lse_theta' in desc)
        return torch.logsumexp(desc['lse_theta'] * L, dim)
