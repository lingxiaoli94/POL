import torch
from torch import nn

class BaseProjectionLayer(nn.Module):
    @property
    def proj_dim(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()


class IdentityProjection(BaseProjectionLayer):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

    @property
    def proj_dim(self):
        return self.input_dim

    def forward(self, x):
        return x

class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, proj_dim):
        super().__init__()
        self._proj_dim = proj_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 2 * proj_dim),
            nn.ReLU(),
            nn.Linear(2 * proj_dim, proj_dim)
        )

    @property
    def proj_dim(self):
        return self._proj_dim

    def forward(self, x):
        return self.proj(x)


class ResidualBlock(nn.Module):
    def __init__(self, *,
                 hidden_size,
                 input_dim,
                 input_proj,
                 feature_proj,
                 use_difference,
                 include_scaling):
        super().__init__()
        input_proj_dim = input_proj.proj_dim
        feature_proj_dim = feature_proj.proj_dim
        self.input_proj = input_proj
        self.feature_proj = feature_proj
        self.use_difference = use_difference
        self.include_scaling = include_scaling

        if include_scaling:
            self.map_s = nn.Sequential(
                nn.Linear(input_proj_dim + feature_proj_dim,
                          hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_dim),
                nn.Hardtanh(min_val=-2, max_val=2)
            )

        self.map_t = nn.Sequential(
            nn.Linear(input_proj_dim + feature_proj_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_dim)
        )

    def forward(self, F, x):
        F_x = torch.cat([self.feature_proj(F), self.input_proj(x)], dim=-1)
        t = self.map_t(F_x)

        if self.use_difference:
            x = x - t
        else:
            x = t
        if self.include_scaling:
            s = self.map_s(F_x)
            x = x * torch.exp(-s)
        return x


class ResidualNet(nn.Module):
    def __init__(self, *,
                 input_dim,
                 feature_dim,
                 input_proj_dim=None,
                 feature_proj_dim=None,
                 num_block,
                 hidden_size,
                 use_difference=True,
                 include_scaling=True):
        super().__init__()

        self.input_proj = (ProjectionLayer(input_dim, input_proj_dim)
                           if input_proj_dim is not None
                           else IdentityProjection(input_dim))

        self.feature_proj = (ProjectionLayer(feature_dim, feature_proj_dim)
                             if feature_proj_dim is not None
                             else IdentityProjection(feature_dim))

        self.blocks = nn.ModuleList()
        for i in range(num_block):
            self.blocks.append(ResidualBlock(
                hidden_size=hidden_size,
                input_dim=input_dim,
                input_proj=self.input_proj,
                feature_proj=self.feature_proj,
                use_difference=use_difference,
                include_scaling=include_scaling,
            ))

    def forward(self, F, x):
        '''
        Args:
            F: NxC
            x: NxBxD
        '''
        if F.ndim == 2:
            B = x.shape[1]
            F = F.unsqueeze(1).expand(-1, B, -1)
        for block in self.blocks:
            x = block(F, x)
        return x
