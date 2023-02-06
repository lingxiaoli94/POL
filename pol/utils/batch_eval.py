import torch
import copy
from tqdm import tqdm


def batch_eval(f, xs, dim=0, batch_size=1024,
               func_device=torch.device('cuda'),
               result_device=torch.device('cpu'),
               detach=True,
               no_tqdm=False):
    total_count = xs.shape[dim]
    result = []
    current_count = 0
    with tqdm(total=total_count, disable=no_tqdm) as pbar:
        while current_count < total_count:
            count = min(batch_size, total_count - current_count)
            inds = [slice(None)] * dim + [slice(current_count, current_count + count), ...]
            cur_result = f(xs[inds].to(func_device))
            if detach:
                cur_result = cur_result.detach()
            result.append(cur_result.to(result_device))
            current_count += count
            pbar.update(count)
        pbar.close()
    return torch.cat(result, dim)

def batch_eval_index(f, total_count, dim, batch_size=1024,
                     result_device=torch.device('cpu'),
                     detach=True,
                     no_tqdm=False):
    result = []
    current_count = 0
    with tqdm(total=total_count, disable=no_tqdm) as pbar:
        while current_count < total_count:
            count = min(batch_size, total_count - current_count)
            inds = slice(current_count, current_count + count)
            cur_result = f(inds)
            if detach:
                cur_result = cur_result.detach()
            result.append(cur_result.to(result_device))
            current_count += count
            pbar.update(count)
        pbar.close()
    return torch.cat(result, dim)
