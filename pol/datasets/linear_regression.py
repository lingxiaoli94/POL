import numpy as np
import torch
from torch.utils.data import TensorDataset


def create_nd_dataset(dim, *,
                      num_sample=1024,
                      seed=1234,
                      noise=0.1,
                      sparsity=1.0,
                      weight_bound=1,
                      torch_type=torch.float32):
    '''
    Args:
        dim: dimension of beta
    '''
    rng = np.random.default_rng(seed)
    W = torch.from_numpy(rng.uniform(-weight_bound, weight_bound, size=[dim]))  # D
    mask = torch.from_numpy(rng.random(size=[dim]) < sparsity)
    while torch.logical_not(mask).all():
        mask = torch.from_numpy(rng.random(size=[dim]) < sparsity)
    W = torch.where(mask, W, torch.zeros_like(W))

    X = torch.from_numpy(rng.normal(scale=1.0, size=[num_sample, dim]))  # NxD
    Y = (X * W).sum(-1, keepdim=True)
    Y = Y + torch.from_numpy(rng.normal(scale=noise, size=Y.shape))  # Nx1

    return TensorDataset(X.to(torch_type), Y.to(torch_type))


def create_toy1_2d_dataset(torch_type=torch.float32):
    '''
    Used in quasinorm lasso to demonstrate multiple global optima.
    This is the same problem as min 0.5 ||XW + W_0 - Y||^2_2 + theta||W||_1 with
    X = I, Y = [1, -1].
    '''
    X = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
    Y = torch.tensor([[1], [-1]])
    return TensorDataset(X.to(torch_type), Y.to(torch_type))


class NormRegularizer:
    def __init__(self, p, eps, stablize=True):
        self.p = p  # p-norm
        self.eps = eps
        self.stablize = stablize

    def create_parameters(self, num_instance, seed, torch_type=torch.float32):
        rng = np.random.default_rng(seed)
        lambdas = torch.from_numpy(
            rng.random([num_instance, 1]))  # Nx1, in [0, 1]

        return lambdas.to(torch_type)

    def regularize_fn(self, W, lambdas):
        # W: ...xD, lambdas: ...x1
        if self.stablize:
            return lambdas.squeeze(-1) * torch.pow((W.square() + self.eps),
                                                             self.p / 2).sum(-1) # ...
        else:
            return lambdas.squeeze(-1) * torch.pow(torch.pow(W.abs() + self.eps,
                                                             self.p).sum(-1),
                                                   1/self.p)  # ...

    def get_lambda_dim(self):
        return 1


class LogSumRegularizer:
    def __init__(self, sigma):
        self.sigma = sigma

    def create_parameters(self, num_instance, seed, torch_type=torch.float32):
        rng = np.random.default_rng(seed)
        lambdas = torch.from_numpy(
            rng.random([num_instance, 1]))  # Nx1, in [0, 1]

        return lambdas.to(torch_type)

    def regularize_fn(self, W, lambdas):
        # W: ...xD, lambdas: ...x1
        return lambdas.squeeze(-1) * torch.log(1 + self.sigma * W.abs()).sum(-1)  # ...

    def get_lambda_dim(self):
        return 1

class NormMixedRegularizer:
    def __init__(self, p_bbox, eps):
        self.p_bbox = p_bbox
        self.eps = eps

    def create_parameters(self, num_instance, seed, torch_type=torch.float32):
        rng = np.random.default_rng(seed)
        lambdas = torch.from_numpy(
            rng.random([num_instance, 1]))  # Nx1, in [0, 1]
        p = torch.from_numpy(
            self.p_bbox[0] + (self.p_bbox[1] - self.p_bbox[0]) * rng.random([num_instance, 1])
        )

        return torch.cat([lambdas.to(torch_type), p.to(torch_type)], -1)

    def regularize_fn(self, W, thetas):
        # W: ...xD, thetas: ...x2
        assert(thetas.shape[-1] == 2)
        lambdas = thetas[..., 0]
        p = thetas[..., 1]
        tmp = torch.pow(W.abs() + self.eps,
                        p.unsqueeze(-1)).sum(-1)
        return lambdas * torch.pow(tmp, 1/p)

    def get_lambda_dim(self):
        return 2


class MCPRegularizer:
    def __init__(self, sigma_range):
        self.sigma_range = sigma_range

    def create_parameters(self, num_instance, seed, torch_type=torch.float32):
        rng = np.random.default_rng(seed)
        sigmas = torch.from_numpy(
            rng.random([num_instance, 1]))  # Nx1, in [0, 1]
        sigmas = self.sigma_range[0] + (
            sigmas * (self.sigma_range[1] - self.sigma_range[0]))

        return sigmas.to(torch_type)


    def regularize_fn(self, W, sigmas):
        # W: ...xD, sigmas: ...x1

        tmp1 = W.abs() - sigmas * W.square()
        tmp2 = 1 / (4 * sigmas)

        tmp = torch.where(W.abs() < 1 / (2 * sigmas),
                          tmp1, tmp2)
        return tmp.sum(-1)


    def get_lambda_dim(self):
        return 1


def make_linear_regression_loss_fn(regularizer):
    '''
    Factory method used to create a loss fn from a regulaizer instance.
    '''
    def loss_fn(X, Y, W, lambdas):
        # X: ...xX, Y: ...x1, W: ...xX, lambdas: ...xC.
        l1 = ((W * X).sum(-1, keepdim=True) - Y).square().squeeze(-1)  # ...
        l2 = regularizer.regularize_fn(W, lambdas)  # ...
        return l1 + l2
    return loss_fn
