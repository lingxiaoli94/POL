import torch
import numpy as np

def is_integral(Y):
    assert(Y.ndim == 1)
    for i in range(Y.shape[0]):
        if not float(Y[i]).is_integer():
            return False
    return True

def filter_cuts(Y, Y_val):
    '''
    Args:
        Y: BxV, 0/1 vectors
        Y_val: B, cut value
    Return:
        Z, Z_loss: KxV, and K, non-repeating cuts
    '''
    B, V = Y.shape

    max_value = int(Y_val.max())

    num_max = 0
    cache = {}
    for b in range(B):
        y = Y[b]
        if not is_integral(y):
            continue
        y = y.astype(int)
        if y[0] == -1:
            y = y * -1
        m = 0
        for i in range(V):
            if y[i] == 1:
                m |= 1 << i
            else:
                assert(y[i] == -1)
        if m not in cache:
            cache[m] = (y, Y_val[b])
            if Y_val[b] == max_value:
                num_max += 1
    num_total = len(cache.keys())
    return {
        'max_prob': 0.0 if num_total == 0 else num_max / num_total,
        'filtered': cache.values()
    }
