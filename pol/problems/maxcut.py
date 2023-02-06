from .problem_base import ProblemBase
import torch
import numpy as np
from torch.utils.data import TensorDataset

class MaxCutProblem(ProblemBase):
    def __init__(self,
                 device,
                 num_vertex,
                 edges,
                 latent_dim,
                 train_thetas,
                 representation,
                 cut_conversion_seed=91,
                 normalize_eps=1e-8,
                 encoder_kind='gcn',
                 encoder_params={}):
        '''
        In this max-cut problem we fix the graph and treat edge weights as
        theta.

        Args:
            num_vertex: V.
            edge_list: Ex2, a tensor representing the undirected edges in the graph.

            train_thetas: NxE, so that train_thetas[i, :] indicate the edge weights
            of the edges in the order of edge_list.

            representation: ['circle', 'angle']
                - For 'angle', X: ...xV
                - For 'circle', X: ...x2V, where first V are x coordinates
        '''
        self.device = device
        self.num_vertex = num_vertex
        self.edges = edges.to(device)
        self.latent_dim = latent_dim
        self.train_thetas = train_thetas
        self.representation = representation
        self.cut_conversion_seed = cut_conversion_seed
        self.normalize_eps = 1e-8

        self.encoder_kind = encoder_kind
        assert(encoder_kind == 'linear') # other encoders are not implemented
        self.encoder = torch.nn.Linear(self.edges.shape[0], latent_dim)
        self.encoder.to(device)

    def set_train_mode(self, is_training):
        if is_training:
            self.encoder.train()
        else:
            self.encoder.eval()

    def get_train_dataset(self):
        return TensorDataset(self.train_thetas)

    def sample_prior(self, data_batch, batch_size):
        '''
        Return: IxBxV (angle) or IxBx2V (circle)
        '''
        I = data_batch[0].shape[0]
        B = batch_size
        V = self.num_vertex

        if self.representation == 'angle':
            a = 2 * np.pi * torch.rand([I, B, V], device=self.device)
            return a
        else:
            assert(self.representation == 'circle')
            P = torch.randn([I, B, 2 * V], device=self.device)
            P = self.apply_projection(data_batch, P)
            return P

    def unpack(self, X):
        '''
        X: ...xV or ...x2V
        Return: ...xVx2, unpacked coordinates on circle
        '''
        V = self.num_vertex
        if self.representation == 'circle':
            P = torch.stack([X[..., :V], X[..., V:]], -1) # ...x2
            P = P / (P.norm(dim=-1, keepdim=True) + self.normalize_eps)
        else:
            P = torch.stack([torch.cos(X), torch.sin(X)], -1)
        return P

    def apply_projection(self, data_batch, X):
        if self.representation == 'angle':
            return torch.sigmoid(X) * 2 * np.pi
        else:
            P = self.unpack(X)
            return torch.cat([P[..., 0], P[..., 1]], -1)

    def push_forward(self, latent_code, X, pushforward_net):
        if self.encoder_kind == 'gcn':
            assert(self.representation == 'circle')
            # HACK: assume pushforward_net has var_dim == 2
            global_feature, vertex_feature = latent_code
            # global_feature: IxC, vertex_feature: IxVxC
            P = self.unpack(X) # IxBxVx2
            I, B, V = P.shape[:-1]
            P = torch.transpose(P, 1, 2).reshape(I * V, B, 2) # IVxBx2
            vertex_feature = vertex_feature.reshape(I * V, -1) # IVxC
            Q = pushforward_net(vertex_feature, P) # IVxBx2
            Q = Q.reshape(I, V, B, 2) # IxVxBx2
            Q = torch.transpose(Q, 1, 2) # IxBxVx2
            return torch.cat([Q[..., 0], Q[..., 1]], -1) # IxBx2V
        else:
            return super().push_forward(latent_code, X, pushforward_net)

    def compute_distance(self, data_batch, X1, X2):
        '''
        X1, X2: ...xV or ...x2V
        '''
        p1 = self.unpack(X1)
        p2 = self.unpack(X2)
        assert(p1.shape[-1] == 2 and p1.shape[-2] == self.num_vertex)
        return (p1 - p2).square().sum(-1).mean(-1)

    def eval_loss(self, data_batch, X):
        '''
        X: IxBxV or IxBx2V
        Return: IxB
        '''
        weights = data_batch[0] # IxE
        P = self.unpack(X)

        # val = 0.5 * weights.unsqueeze(-2) * torch.cos(
        #     X[:, :, self.edges[:, 0]] - X[:, :, self.edges[:, 1]]) # IxBxE
        cos_diff = (P[:, :, self.edges[:, 0], :] * P[:, :, self.edges[:, 1], :]).sum(-1) # IxBxE
        val = 0.5 * weights.unsqueeze(-2) * cos_diff # IxBxE
        return val.sum(-1) # positive since minimization

    def extract_latent_code(self, data_batch):
        return self.encoder(data_batch[0])

    def get_nn_parameters(self):
        return self.encoder.parameters()

    def get_state_dict(self):
        return self.encoder.state_dict()

    def load_state_dict(self, d):
        self.encoder.load_state_dict(d)

    def convert_to_cut(self, weights, X, rand_order=True):
        '''
        weights: IxE
        X: IxBxV or IxBx2V
        Return: (IxBxV, IxB), best cut and best objective value.
        The returning cut will be represented with -1 or +1 for each vertex.

        Convert an angle representation to a cut, following
        a Geomans-Williamson-type strategy.
        '''

        gen = torch.Generator(device=X.device)
        gen.manual_seed(self.cut_conversion_seed)

        P = self.unpack(X) # IxBxVx2
        I, B = P.shape[:2]
        V = self.num_vertex
        Y_best = torch.zeros([I, B, V]).to(P) # IxBxV
        val_best = torch.full([I, B], -1e10).to(P) # IxB
        if rand_order:
            order = torch.argsort(torch.rand([I, B, V], device=X.device,
                                             generator=gen),
                                  dim=-1) # IxBxV
        else:
            order = torch.arange(
                self.num_vertex,
                device=X.device)[None, None, :].expand(I, B, -1)
        p_reordered = torch.stack([
            torch.gather(P[..., 0], dim=-1, index=order),
            torch.gather(P[..., 1], dim=-1, index=order)
        ], -1) # IxBxVx2
        for i in range(self.num_vertex):
            p_lb = P[:, :, i, :] # IxBx2
            cross_prod = (p_lb[:, :, 0].unsqueeze(-1) * P[:, :, :, 1] -
                          p_lb[:, :, 1].unsqueeze(-1) * P[:, :, :, 0]) # IxBxV
            Y = torch.where(cross_prod >= 0,
                            torch.full_like(cross_prod, 1.0),
                            torch.full_like(cross_prod, -1.0)) # IxBxV
            val = 0.5 * weights.unsqueeze(-2) * (
                1 - Y[:, :, self.edges[:, 0]] * Y[:, :, self.edges[:, 1]]) # IxBxE
            val = val.sum(-1) # IxB

            tmp = (val > val_best).unsqueeze(-1).expand(-1, -1, self.num_vertex) # IxBxV
            Y_best = torch.where(tmp, Y, Y_best)
            val_best = torch.where(val > val_best, val, val_best)

        return (Y_best, val_best)


    def validate(self, solver, aux_data=None):
        from pol.utils.validation.shortcuts import simple_validate

        def info_fn(thetas):
            result = {
                'type': 'ndarray_dict',
                'edges': self.edges.detach().cpu(),
                'thetas': thetas.detach().cpu()
            }
            if aux_data.get('include_witness', False):
                for i in range(10):
                    witness = self.sample_prior([thetas],
                                                aux_data.get('witness_batch_size', 1024))
                    result[f'witness_{i}'] = witness.detach().cpu()

            return result

        def result_fn(thetas, X):
            Y, val = self.convert_to_cut(thetas, X)
            result = {
                'type': 'ndarray_dict',
                'X': X.detach().cpu(),
                'Y': Y.detach().cpu()
            }
            if aux_data.get('include_loss', True):
                result['X_loss'] = self.eval_loss([thetas], X).detach().cpu()
                result['Y_loss'] = val.detach().cpu()
                result['satisfy'] = self.do_satisfy([thetas], X).detach().cpu()

            return result

        return simple_validate(self, solver, aux_data, info_fn, result_fn)
