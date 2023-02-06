import torch
import numpy as np


def create_linear_layers(in_dim, out_dim, hidden_layer_sizes):
    return torch.nn.ModuleList([
        torch.nn.Linear(in_dim if i == 0 else hidden_layer_sizes[i - 1],
                        out_dim if i == len(hidden_layer_sizes)
                        else hidden_layer_sizes[i])
        for i in range(len(hidden_layer_sizes) + 1)
    ])


def create_activation_fn(activation):
    if activation == 'relu':
        return torch.relu
    elif activation == 'tanh':
        return torch.tanh
    elif activation == 'celu':
        return torch.celu
    elif activation == 'leaky_relu':
        return torch.nn.functional.leaky_relu
    elif activation == 'softplus':
        return torch.nn.functional.softplus
    else:
        raise Exception('Activation option {} is unknown'.format(activation))


def forward_linear_layers(x, linear_layers, activation_fn):
    for i, layer in enumerate(linear_layers):
        x = layer(x)
        if i < len(linear_layers) - 1:
            x = activation_fn(x)
    return x


class MLP(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 hidden_layer_sizes,
                 activation='relu'):
        super().__init__()
        self.layers = create_linear_layers(in_dim, out_dim, hidden_layer_sizes)
        self.activation_fn = create_activation_fn(activation)

    def forward(self, x):
        return forward_linear_layers(x, self.layers, self.activation_fn)


class ConditionalMLP(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 feature_dim,
                 out_dim,
                 hidden_layer_sizes,
                 activation='relu',
                 use_residual=False,
                 rff_first=False,
                 rff_dim=128,
                 rff_sigma=2.0,
                 proj_first=False,
                 proj_dim=128):
        super().__init__()
        self.use_residual = use_residual
        self.rff_first = rff_first
        self.proj_first = proj_first
        if rff_first:
            self.register_buffer('rff', torch.randn(rff_dim // 2, in_dim) * rff_sigma)
            combined_in_dim = rff_dim + feature_dim
        elif proj_first:
            from .simple_nvp import ProjectionLayer
            self.proj_layer = ProjectionLayer(in_dim, proj_dim)
            combined_in_dim = proj_dim + feature_dim
        else:
            combined_in_dim = in_dim + feature_dim
        self.layers = create_linear_layers(combined_in_dim, out_dim, hidden_layer_sizes)
        self.activation_fn = create_activation_fn(activation)

        # def init_weights(m):
        #     if isinstance(m, torch.nn.Linear):
        #         m.weight.data.normal_(0.0, 0.002)
        #         m.bias.data.fill_(0)
        # self.apply(init_weights)

    def forward(self, F, x):
        # F - NxF
        # x - NxBxI
        x_in = x
        if self.rff_first:
            x = torch.matmul(2 * np.pi * x, self.rff.T)
            x = torch.cat([torch.sin(x), torch.cos(x)], -1)
        elif self.proj_first:
            x = self.proj_layer(x)
        x = torch.cat([x, F.unsqueeze(1).repeat(1, x.shape[1], 1)], -1)
        x_out = forward_linear_layers(x, self.layers, self.activation_fn)
        if self.use_residual:
            x_out = x_out + x_in
        return x_out


class DeterministicConditionalMLP(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 feature_dim,
                 out_dim,
                 hidden_layer_sizes,
                 activation='relu'):
        super().__init__()
        self.layers = create_linear_layers(feature_dim, out_dim, hidden_layer_sizes)
        self.activation_fn = create_activation_fn(activation)

    def forward(self, F, x):
        # Maps feature directly to output.
        # F - NxF
        # x - NxBxI
        return forward_linear_layers(F.unsqueeze(1).repeat(1, x.shape[1], 1),
                                     self.layers, self.activation_fn)
