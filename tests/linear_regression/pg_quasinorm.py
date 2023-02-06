import torch
import numpy as np
from tqdm import tqdm
import random
from pathlib import Path
import h5py
import pol
import pol.datasets.linear_regression
from pol.datasets.linear_regression import NormRegularizer
from pol.utils.validation.scene_saver import save_scenes

def prox_quasinorm(x, l):
    '''
    Solve min_y ||x-y||_2^2 + l ||y||_1/2^{1/2} using
    Fast image deconvolution using closed-form thresholding formulas of $l_p (p=1/2, 2/3).

    x: BxD
    l: B
    '''
    B, D = x.shape
    x = x.reshape(-1) # -1
    l = l.unsqueeze(-1).expand(-1, D).reshape(-1) # -1

    x_abs = x.abs()
    phi = torch.acos(l / 8 * torch.pow(x_abs / 3, -3/2)) # B
    p = np.power(54, 1/3.) / 4 * torch.pow(l, 2/3) # B

    res1 = 2/3 * x_abs * (1 + torch.cos(2 * np.pi / 3 - 2 * phi / 3))
    res2 = torch.zeros_like(x)
    res3 = -res1

    res = torch.where(x > p, res1,
                      torch.where(x > -p, res2, res3))
    res = res.reshape(B, D)
    return res


def test(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    B = 1000
    D = 8
    for i in tqdm(range(100)):
        x = torch.randn([B, D])
        l = torch.rand([B])
        y_pred = prox_quasinorm(x, l)
        y_rand = torch.randn([B, D])

        def eval(y):
            return (y - x).square().sum(-1) + l * torch.pow(y.abs(), 1/2).sum(-1)
        v_pred = eval(y_pred)
        v_rand = eval(y_rand)
        if (not (v_pred < v_rand + 1e-8).all()):
            mask = v_pred > v_rand + 1e-8
            print('pred: ', v_pred[mask])
            print('rand: ', v_rand[mask])
            print('x: ', x[mask])
            print('l: ', l[mask])
            print('y_pred: ', y_pred[mask])
            print('y_rand: ', y_rand[mask])
            break


if __name__ == '__main__':
    # test()

    dataset = pol.datasets.linear_regression.create_nd_dataset(dim=8,
                                                               num_sample=4,
                                                               seed=1234)
    regularizer = NormRegularizer(p=0.5, eps=1e-8)
    theta = regularizer.create_parameters(num_instance=128, seed=9998) # need to match the one in config

    device = torch.device('cuda')
    # device = torch.device('cpu')
    X, Y = dataset[:]
    X = X.to(device)
    Y = Y.to(device)
    alpha = theta.squeeze(-1).to(device) # M
    # X - NxD, Y - Nx1

    B = 4096 # particles per alpha
    M = theta.shape[0]
    num_itr = 200
    lr = 4e-2 # 1e-1 explodes
    h5_dir = Path('./scenes/quasinorm_8d/pg')
    if not h5_dir.exists():
        h5_dir.mkdir(parents=True, exist_ok=True)

    N, D = X.shape
    W = torch.rand([B * M, D], device=device) # (BM) x D
    alpha = alpha.unsqueeze(0).expand(B, -1).reshape(-1) # BM

    def loss_fn(W):
        tmp = (X.unsqueeze(0) @ W.unsqueeze(-1) - Y).square().sum(-1).sum(-1) # BM
        tmp = tmp + alpha * torch.pow(W.abs(), 1/2).sum(-1)
        return tmp


    save_itrs = list(range(0, num_itr, 1))
    scene = {}
    for itr in tqdm(range(num_itr)):
        if itr in save_itrs:
            loss = loss_fn(W) # BM
            # print(loss.reshape(B, M).mean(0))
            satisfy = torch.logical_and(W <= 2, W >= -2).all(-1) # BM
            scene[f'itr_{itr}'] = {
                'W': W.reshape(B, M, D).transpose(0, 1).cpu(),
                'loss': loss.reshape(B, M).transpose(0, 1).cpu(),
                'satisfy': satisfy.reshape(B, M).transpose(0, 1).cpu(),
                'type': 'ndarray_dict'}


        grad_F = 2 * X.t().unsqueeze(0) @ (X.unsqueeze(0) @ W.unsqueeze(-1) - Y) # BxDx1
        grad_F = grad_F.squeeze(-1) # BxD

        tmp = W - lr * grad_F # BxD
        W = prox_quasinorm(tmp, 2 * lr * alpha)

    save_scenes([scene], h5_dir / 'result.h5')
