import numpy as np
import torch
from abc import ABC, abstractmethod
from torch.utils.data import TensorDataset
from pol.utils.parsing import list_to_tensor


class AnalyticalBase(ABC):
    @abstractmethod
    def eval_fn(self, X, theta):
        pass

    @abstractmethod
    def get_var_bbox(self):
        # Return Dx2
        pass

    def change_device(self, device):
        pass

    @abstractmethod
    def create_parameters(self, num_instance, seed):
        pass


class ProductSin(AnalyticalBase):
    def __init__(self, dim, var_bbox):
        self.dim = dim
        self.var_bbox = list_to_tensor(var_bbox)

    def get_var_bbox(self):
        return self.var_bbox

    def eval_fn(self, X, theta):
        # X: ...xD, theta: ...xD
        tmp = torch.sin(theta * X)
        return torch.prod(tmp, -1)  # ...

    def create_parameters(self, num_instance, seed):
        rng = np.random.default_rng(seed)
        return torch.from_numpy(0.2 + 0.8 * rng.random([num_instance, self.dim])).to(torch.float32) # IxC

class ConicSection(AnalyticalBase):
    '''
    Ax^2+Bxy+Cy^2+Dx+Ey+F=0
    '''
    def __init__(self, var_bbox, ellipse_only):
        self.var_bbox = list_to_tensor(var_bbox)
        self.ellipse_only = ellipse_only

    def get_var_bbox(self):
        return self.var_bbox

    def eval_fn(self, X, theta):
        # X: ...x2, theta: ...x6
        A = theta[..., 0]
        B = theta[..., 1]
        C = theta[..., 2]
        D = theta[..., 3]
        E = theta[..., 4]
        F = theta[..., 5]

        x = X[..., 0]
        y = X[..., 1]

        f = (A * x * x + B * x * y + C * y * y +
                D * x + E * y + F)
        return f.square()

    def create_parameters(self, num_instance, seed):
        rng = np.random.default_rng(seed)
        if self.ellipse_only:
            parameters = []
            for i in range(num_instance):
                while True:
                    param = torch.from_numpy(2 * rng.random([6]) - 1).to(torch.float32)
                    A, B, C = param[:3]
                    if B * B < 4 * A * C:
                        break
                parameters.append(param)
            return torch.stack(parameters, 0)
        else:
            return torch.from_numpy(2 * rng.random([num_instance, 6]) - 1).to(torch.float32)

class L1Norm(AnalyticalBase):
    def __init__(self, var_bbox):
        self.var_bbox = list_to_tensor(var_bbox)

    def get_var_bbox(self):
        return self.var_bbox

    def eval_fn(self, X, theta):
        # X: ...xD, theta: ...x1
        return theta.squeeze(-1) * X.abs().sum(-1)

    def create_parameters(self, num_instance, seed):
        return torch.ones([num_instance, 1], dtype=torch.float32)


class Rastrigin(AnalyticalBase):
    def __init__(self, dim, var_bbox):
        self.dim = dim
        self.var_bbox = list_to_tensor(var_bbox)

    @staticmethod
    def pack_rastrigin(A, b, c):
        # A: ...xDxD, b: ...xD, c: ...xD
        D = b.shape[-1]
        lead_ndim = b.dim() - 1
        return torch.cat([
            A.reshape(list(b.shape[:-1]) + [D*D]),
            b, c
        ])  # ...xD(D+2)

    def eval_fn(self, X, theta):
        # X: ...xD, theta: ...xD(D+2)
        lead_ndim = X.dim() - 1
        D = X.shape[-1]
        assert(D == self.dim)
        assert(theta.shape[-1] == D*(D+2))
        A = theta[..., :D*D].reshape(list(X.shape[:-1]) + [D, D])  # ...XDxD
        b = theta[..., D*D:D*(D+1)]  # ...xD
        c = theta[..., D*(D+1):D*(D+2)]  # ...xD

        tmp1 = 0.5 * (torch.matmul(A, X.unsqueeze(-1)).squeeze(-1) - b).square().sum(-1)  # ...
        tmp2 = -10 * (c * torch.cos(2 * np.pi * X)).sum(-1) + 10 * D  # ...

        return tmp1 + tmp2  # ...

    def get_var_bbox(self):
        return self.var_bbox

    def create_parameters(self, num_instance, seed):
        if seed == 0:
            A = torch.zeros([self.dim, self.dim])
        else:
            A = torch.eye(self.dim)
        b = torch.zeros([self.dim])
        c = torch.ones([self.dim])

        theta = self.pack_rastrigin(A, b, c).to(torch.float32)
        return theta.unsqueeze(0)  # 1xD(D+2)


def make_analytical_loss_fn(prob):
    def loss_fn(X, theta):
        return prob.eval_fn(X, theta)
    return loss_fn
