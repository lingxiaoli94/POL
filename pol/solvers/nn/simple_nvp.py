# Modifed from https://github.com/paschalidoud/neural_parts/blob/master/neural_parts/models/simple_nvp.py
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import logging
from tqdm import tqdm


class BaseProjectionLayer(nn.Module):
    @property
    def proj_dim(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()


class IdentityProjection(BaseProjectionLayer):
    def __init__(self, input_dim):
        super().__init__()
        self._input_dim = input_dim

    @property
    def proj_dim(self):
        return self._input_dim

    def forward(self, x):
        return x


class ProjectionLayer(BaseProjectionLayer):
    def __init__(self, input_dim, proj_dim):
        super().__init__()
        self._proj_dim = proj_dim

        self.proj = nn.Sequential(
            nn.Linear(input_dim, 2*proj_dim),
            nn.ReLU(),
            nn.Linear(2*proj_dim, proj_dim)
        )

    @property
    def proj_dim(self):
        return self._proj_dim

    def forward(self, x):
        return self.proj(x)


class CouplingLayer(nn.Module):
    def __init__(self, map_s, map_t, projection, feature_projection, mask,
                 no_mask, no_scaling):
        super().__init__()
        self.map_s = map_s
        self.map_t = map_t
        self.projection = projection
        self.feature_projection = feature_projection
        self.register_buffer("mask", mask)
        self.no_mask = no_mask
        self.no_scaling = no_scaling

    def forward(self, F, y):
        y1 = y * self.mask

        if self.no_mask:
            F_y1 = torch.cat([self.feature_projection(F), self.projection(y)], dim=-1)
        else:
            F_y1 = torch.cat([self.feature_projection(F), self.projection(y1)], dim=-1)
        if self.no_scaling:
            s = torch.zeros_like(y)
        else:
            s = self.map_s(F_y1)
        t = self.map_t(F_y1)

        x = y1 + (1-self.mask) * ((y - t) * torch.exp(-s))

        return x

    def inverse(self, F, x):
        assert(not self.no_mask)
        x1 = x * self.mask

        F_x1 = torch.cat([self.feature_projection(F), self.projection(x1)], dim=-1)
        s = self.map_s(F_x1)
        t = self.map_t(F_x1)

        y = x1 + (1-self.mask) * (x * torch.exp(s) + t)

        return y


class SimpleNVP(nn.Module):
    def __init__(self,
                 num_layers,
                 feature_dim,
                 hidden_size,
                 projection_dim,
                 input_dim=3,
                 checkpoint=False,
                 normalize=False,
                 project_feature=False,
                 no_mask=False,
                 no_scaling=False):
        super().__init__()
        self._input_dim = input_dim
        self._checkpoint = checkpoint
        self._normalize = normalize
        self._projection = ProjectionLayer(input_dim, projection_dim)
        self.no_mask = no_mask
        self.no_scaling = no_scaling
        if project_feature:
            self._feature_projection = ProjectionLayer(feature_dim, projection_dim)
            feature_proj_dim = projection_dim
        else:
            self._feature_projection = nn.Identity()
            feature_proj_dim = feature_dim
        self._create_layers(num_layers, feature_dim, feature_proj_dim, hidden_size)

    def _create_layers(self, num_layers, feature_dim, feature_proj_dim, hidden_size):
        input_dim = self._input_dim
        proj_dim = self._projection.proj_dim

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # mask = torch.zeros(input_dim)
            # # TODO: change this
            # perm = torch.randperm(input_dim)[:2]
            # mask[perm] = 1
            if self.no_mask:
                mask = torch.zeros(input_dim)
            else:
                mask = torch.ones(input_dim)
                mask[i % input_dim] = 0

            map_s = nn.Sequential(nn.Linear(proj_dim+feature_proj_dim,
                                            hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size, hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size, input_dim),
                                  nn.Hardtanh(min_val=-2, max_val=2)
                                  )
            map_t = nn.Sequential(
                nn.Linear(proj_dim+feature_proj_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_dim)
            )
            self.layers.append(CouplingLayer(
                map_s,
                map_t,
                self._projection,
                self._feature_projection,
                mask,
                no_mask=self.no_mask,
                no_scaling=self.no_scaling,
            ))

        if self._normalize:
            self.scales = nn.Sequential(
                nn.Linear(feature_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_dim)
            )

    def _check_shapes(self, F, x):
        if F.ndim == 3:
            assert(F.shape[0] == x.shape[0] and F.shape[1] == x.shape[1])
        else:
            assert(F.shape[0] == x.shape[0])

    def _expand_features(self, F, x):
        _, B, _ = x.shape
        return F[:, None, :].expand(-1, B, -1)  # NxBxZ

    def _call(self, func, *args, **kwargs):
        if self._checkpoint:
            return checkpoint(func, *args, **kwargs)
        else:
            return func(*args, **kwargs)

    def _normalize_input(self, F, y):
        if not self._normalize:
            return 0, 1

        sigma = torch.nn.functional.elu(self.scales(F)) + 1  # NxD
        sigma = sigma[:, None, :]  # Nx1xD

        return 0, sigma

    def forward(self, F, x):
        self._check_shapes(F, x)
        mu, sigma = self._normalize_input(F, x)
        if F.ndim == 2:
            F = self._expand_features(F, x)

        y = x
        for l in self.layers:
            y = self._call(l, F, y)
        y = y / sigma + mu
        return y

    def inverse(self, F, y):
        self._check_shapes(F, y)
        mu, sigma = self._normalize_input(F, y)
        if F.ndim == 2:
            F = self._expand_features(F, y)

        x = y
        x = (x - mu) * sigma
        for l in reversed(self.layers):
            x = self._call(l.inverse, F, x)
        return x


class SimpleNVPSingle(nn.Module):
    def __init__(self,
                 num_layers,
                 hidden_size,
                 projection_dim,
                 input_dim=3,
                 checkpoint=True,
                 normalize=True):
        super().__init__()
        self.simple_nvp = SimpleNVP(
            num_layers=num_layers,
            feature_dim=1,
            hidden_size=hidden_size,
            projection_dim=projection_dim,
            input_dim=input_dim,
            checkpoint=checkpoint,
            normalize=normalize
        )

    def forward(self, x):
        return self.simple_nvp.forward(torch.zeros([1, 1]).to(x), x.unsqueeze(0)).squeeze(0)

    def inverse(self, F, y):
        return self.simple_nvp.inverse(torch.zeros([1, 1]).to(y), y.unsqueeze(0)).squeeze(0)


class DeterministicSimpleNVP(nn.Module):
    def __init__(self,
                 num_layers,
                 feature_dim,
                 hidden_size,
                 projection_dim,
                 input_dim=3,
                 checkpoint=True,
                 normalize=True):
        super().__init__()
        self.simple_nvp = SimpleNVP(
            num_layers=num_layers,
            feature_dim=1,
            hidden_size=hidden_size,
            projection_dim=projection_dim,
            input_dim=feature_dim,
            checkpoint=checkpoint,
            normalize=normalize
        )

    def forward(self, F, x):
        # Return y that is determined by F entirely.
        assert(F.shape[-1] == x.shape[-1])
        return self.simple_nvp.forward(torch.zeros([F.shape[0], 1]).to(F),
                                       F.unsqueeze(1).repeat(1, x.shape[1], 1))
