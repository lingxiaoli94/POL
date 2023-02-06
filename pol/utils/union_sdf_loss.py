import torch


def compute_proxy_union_sdf(union_sdf_vals):
    proxy_val, _ = torch.stack(union_sdf_vals, -1).min(-1)
    return proxy_val


def compute_proxy_target_loss(proxy_val, target_sdf_val):
    # union_sdf_vals is a list of equally sized 1D tensors.
    E_plus = torch.where(target_sdf_val >= 0,
                         (proxy_val - target_sdf_val).square(),
                         torch.zeros_like(proxy_val))
    E_minus = torch.where(torch.logical_and(target_sdf_val < 0, proxy_val > 0),
                          proxy_val.square(),
                          torch.zeros_like(proxy_val))
    return (E_plus, E_minus)
