import torch
import numpy as np
from tqdm import tqdm
import random
from pathlib import Path
import h5py
import pol
import pol.datasets.linear_regression
from pol.datasets.linear_regression import MCPRegularizer
from pol.utils.validation.scene_saver import save_scenes


def reg_fn(y, sigma):
    result = torch.where(y.abs() <= 1/(2*sigma),
                         y.abs() - sigma * y.square(),
                         1/(4*sigma))
    return result


def eval_fn(y, x, sigma, eta):
    return (y - x).square() + eta * reg_fn(y, sigma)


def prox_mcp(x, sigma, eta):
    '''
    Solve min_y ||x-y||_2^2 + eta * R_mcp(y; sigma),
    where R_mcp(y; sigma) = |y| - sigma * y^2 if y <= 1/2sigma and 1/4sigma otherwise.

    x: BxD
    sigma: B
    eta: scalar
    '''
    B, D = x.shape
    x = x.reshape(-1) # -1
    sigma = sigma.unsqueeze(-1).expand(-1, D).reshape(-1) # -1


    x_abs = x.abs()
    res1 = x
    res2 = 1 / (2 * sigma)
    res3 = (2 * x - eta) / (2 - 2 * eta * sigma)
    res4 = (2 * x + eta) / (2 - 2 * eta * sigma)
    res5 = torch.zeros_like(x)

    res_all = torch.stack([res1, res2, res3, res4, res5], -1)
    val_stacked = eval_fn(res_all, x.unsqueeze(-1), sigma.unsqueeze(-1), eta)
    best_val, best_id = val_stacked.min(-1)

    res = res_all[torch.arange(B*D), best_id]

    return res.reshape(B, D), best_val.reshape(B, D)


def test():
    B = 100
    D = 1
    eps = 1e-6
    for i in tqdm(range(10000)):
        x = torch.randn([B, D])
        sigma = torch.rand([B])
        eta = torch.rand([1]).item()
        sigma_ex = sigma.unsqueeze(-1)
        y_pred, best_pred = prox_mcp(x, sigma, eta)
        y_rand = torch.randn([B, D])

        def eval(y):
            return eval_fn(y, x, sigma_ex, eta).sum(-1)
        v_pred = eval(y_pred)
        v_rand = eval(y_rand)
        if (not (v_pred <= v_rand + eps).all()):
            mask = v_pred > v_rand + eps
            print('v pred: ', v_pred[mask])
            print('v rand: ', v_rand[mask])
            print('best pred: ', best_pred[mask])
            print('x: ', x[mask])
            print('sigma: ', sigma[mask])
            print('y_pred: ', y_pred[mask])
            print('y_rand: ', y_rand[mask])
            break


if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # test()

    dataset = pol.datasets.linear_regression.create_nd_dataset(dim=8,
                                                               num_sample=4,
                                                               seed=1234)
    regularizer = MCPRegularizer(sigma_range=[0.5, 2.0])
    theta = regularizer.create_parameters(num_instance=128, seed=9999) # need to match the one in config

    device = torch.device('cuda')
    # device = torch.device('cpu')
    X, Y = dataset[:]
    X = X.to(device)
    Y = Y.to(device)
    sigma = theta.squeeze(-1).to(device) # M
    # X - NxD, Y - Nx1

    B = 4096 # particles per sigma
    M = theta.shape[0]
    num_itr = 20000
    lr = 1e-4 # 1e-1 explodes
    h5_dir = Path('./scenes/mcp_8d/pg')
    if not h5_dir.exists():
        h5_dir.mkdir(parents=True, exist_ok=True)

    N, D = X.shape
    W = torch.rand([B * M, D], device=device) # (BM) x D
    sigma = sigma.unsqueeze(0).expand(B, -1).reshape(-1) # BM

    def loss_fn(W):
        assert((sigma >= 0).all())
        tmp = (X.unsqueeze(0) @ W.unsqueeze(-1) - Y).square().sum(-1).sum(-1) # BM
        assert((tmp >= 0).all())
        reg_tmp = reg_fn(W, sigma.unsqueeze(-1).expand(-1, D)).sum(-1)
        assert((reg_tmp >= 0).all())
        tmp = tmp + reg_tmp
        return tmp


    save_itrs = [num_itr - 1] # list(range(0, num_itr, 1))
    scene = {}
    for itr in tqdm(range(num_itr)):
        if itr in save_itrs:
            loss = loss_fn(W) # BM
            print('loss: ', loss)
            satisfy = torch.logical_and(W <= 2, W >= -2).all(-1) # BM
            scene[f'itr_{itr}'] = {
                'W': W.reshape(B, M, D).transpose(0, 1).cpu(),
                'loss': loss.reshape(B, M).transpose(0, 1).cpu(),
                'satisfy': satisfy.reshape(B, M).transpose(0, 1).cpu(),
                'type': 'ndarray_dict'}


        grad_F = 2 * X.t().unsqueeze(0) @ (X.unsqueeze(0) @ W.unsqueeze(-1) - Y) # BxDx1
        grad_F = grad_F.squeeze(-1) # BxD

        tmp = W - lr * grad_F # BxD
        W = prox_mcp(tmp, sigma, 2 * lr)[0]

    save_scenes([scene], h5_dir / 'result.h5')
