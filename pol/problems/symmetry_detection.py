import torch
import numpy as np
from torch.utils.data import TensorDataset
from .problem_base import ProblemBase
from pol.utils.reflection import apply_reflection
from pol.datasets.mesh_repository import MeshRepository
from pol.solvers.nn.pointnet import PointNetEncoder
from pol.solvers.nn.dgcnn import DGCNNEncoder
from tqdm import tqdm


class SymmetryDetection(ProblemBase):
    def __init__(self,
                 *,
                 device,
                 train_repo,
                 latent_dim=1024,
                 normal_weight=0.0,
                 normal_loss_exponent=10.0,
                 normalize_eps=1e-8,
                 squared_distance=True,
                 loss_aggregate_desc={'mode': 'mean'},
                 encoder_kind='pointnet',
                 encoder_params={},
                 use_normal_feature=True,
                 zero_latent_code=False):
        assert(device == torch.device('cuda'))
        self.device = device
        self.train_repo = train_repo
        self.latent_dim = latent_dim
        self.normal_weight = normal_weight
        self.normal_loss_exponent = normal_loss_exponent
        self.use_normal_feature = use_normal_feature
        self.zero_latent_code = zero_latent_code
        self.num_shape = self.train_repo.get_num_mesh()
        self.normalize_eps = normalize_eps
        self.squared_distance = squared_distance
        self.loss_aggregate_desc = loss_aggregate_desc

        num_in_channel = 3 if not use_normal_feature else 6
        if encoder_kind == 'pointnet':
            self.encoder = PointNetEncoder(channel=num_in_channel,
                                           latent_dim=latent_dim,
                                           **encoder_params)
        else:
            assert(encoder_kind == 'dgcnn')
            self.encoder = DGCNNEncoder(input_channels=num_in_channel,
                                        output_channels=latent_dim,
                                        **encoder_params)
        self.encoder.to(device)

    def set_train_mode(self, is_training):
        if is_training:
            self.encoder.train()
        else:
            self.encoder.eval()

    def get_train_dataset(self):
        return TensorDataset(torch.arange(self.num_shape))

    def sample_prior(self, data_batch, batch_size):
        '''
        Return: IxBxD, D = 4
        A reflection is parameterized as [n, d], where n is the unit normal of
        the rotation plane and d is the intercept.
        The intercept is assumed to be in [0, 1].
        '''
        index_batch = data_batch[0]

        N = torch.randn([index_batch.shape[0], batch_size, 3], device=self.device)
        I = 0.2 * torch.randn([index_batch.shape[0], batch_size, 1], device=self.device)

        X = torch.cat([N, I], -1)
        return self.apply_projection(data_batch, X)

    def apply_projection(self, data_batch, X):
        N = X[:, :, :3]
        N = N / (N.norm(dim=-1, keepdim=True) + self.normalize_eps)
        I = X[:, :, 3].unsqueeze(-1)
        I = torch.abs(I)
        R = torch.cat([N, I], -1)  # IxB2x4
        return R

    def compute_normal_loss(self, N1, N2):
        '''
        N1, N2: ...x3, sdf: ...
        Return: ...
        '''
        # For now assume the normals are not oriented.
        loss = ((N1 * N2).sum(-1).abs() - 1).abs()
        return loss

    def eval_repo_loss(self, repo, data_batch, X, separate_normal=False):
        '''
        X: IxB2xD
        Return: IxB2, fidelity loss
        '''
        assert(X.shape[2] == 4)
        I = X.shape[0]

        index_batch = data_batch[0]
        PN = repo.get_fixed_pcs(index_batch, include_normal=True)  # IxB1x6

        P = PN[..., :3] # IxB1x3
        N = PN[..., 3:] # IxB1x3
        P_reflected = apply_reflection(P, X)  # IxB1xB2x3
        N_reflected = apply_reflection(N, X)  # IxB1xB2x3

        sdfs, inds = repo.compute_sdf(
            index_batch, P_reflected.reshape(I, -1, 3))
        # sdfs, inds: IxB1B2

        sdfs = sdfs.reshape(P_reflected.shape[:-1])  # IxB1xB2

        target_normals = repo.get_normals(index_batch, inds) # IxB1B2x3
        target_normals = target_normals.reshape(N_reflected.shape)  # IxB1xB2x3

        if self.squared_distance:
            loss = sdfs.square() # IxB1xB2
        else:
            loss = sdfs.abs() # IxB1xB2

        normal_loss = self.compute_normal_loss(
            N_reflected, target_normals)

        from pol.utils.loss_aggregation import aggregate
        if separate_normal:
            normal_loss = aggregate(normal_loss, 1, self.loss_aggregate_desc)
        else:
            normal_loss = normal_loss * torch.exp(-self.normal_loss_exponent * sdfs.detach().abs())
            loss += self.normal_weight * normal_loss

        loss = aggregate(loss, 1, self.loss_aggregate_desc)  # IxB2

        if separate_normal:
            return loss, normal_loss
        else:
            return loss

    def extract_repo_latent_code(self, repo, data_batch):
        '''
        data_batch : I
        Return: IxC
        '''
        index_batch = data_batch[0]
        if self.zero_latent_code:
            return torch.zeros([index_batch.shape[0], self.latent_dim], dtype=torch.float32).to(self.device)
        P = repo.get_fixed_pcs(index_batch,
                               include_normal=self.use_normal_feature)  # IxB1x3
        return self.encoder(P)  # IxC

    def eval_loss(self, data_batch, X):
        return self.eval_repo_loss(self.train_repo, data_batch, X)

    def extract_latent_code(self, data_batch):
        return self.extract_repo_latent_code(self.train_repo, data_batch)

    def get_nn_parameters(self):
        return self.encoder.parameters()

    def get_state_dict(self):
        return self.encoder.state_dict()

    def load_state_dict(self, d):
        self.encoder.load_state_dict(d)

    def validate(self, solver, aux_data):
        '''
        Args:
            solver: a solver instance
            aux_data: dict containing:
                * test_repo
                * prior_batch_size
                * num_itr
                * var_dict
        '''
        test_repo = aux_data['test_repo']
        num_mesh = test_repo.get_num_mesh()

        if solver.is_universal():
            num_itr = aux_data['num_itr']
        else:
            X_history = solver.extract_solutions()
            # assert(X_history[0].shape[0] == num_mesh)
            num_itr = len(X_history)

        # One scene per test shape.
        scenes = []
        total_size = 0
        total_loss = 0
        total_normal_loss = 0
        for i in tqdm(range(num_mesh)):
            scene = {}
            mesh = test_repo.get_mesh_lazy(i)
            scene['mesh'] = {
                'type': 'ndarray_dict',
                'vertices': mesh.vertices,
                'faces': mesh.faces
            }

            index_batch = torch.tensor([i])
            data_batch = [index_batch]
            if solver.is_universal():
                latent_code = self.extract_repo_latent_code(test_repo, data_batch).detach()
                prior_samples = self.sample_prior(
                    data_batch, aux_data['prior_batch_size'])

                X = prior_samples
                Xs = []
                Xs.append(X.cpu().detach())
                for j in range(num_itr):
                    X = solver.push_forward(data_batch, latent_code, X.detach())
                    Xs.append(X.cpu().detach())
            else:
                Xs = []
                for j in range(num_itr):
                    Xs.append(X_history[j][i].unsqueeze(0).detach().clone())

            for j in range(num_itr):
                X = Xs[j] # 1xB2x4
                L, nL = self.eval_repo_loss(
                    test_repo, data_batch, X.to(self.device), separate_normal=True)
                L = L.detach().cpu() # 1xB2
                nL = nL.detach().cpu() # 1xB2
                X = X.squeeze(0)
                L = L.squeeze(0)
                nL = nL.squeeze(0)
                scene[f'itr_{j}'] = {
                    'type': 'ndarray_dict',
                    'reflections': X,
                    'losses': L,
                    'normal_losses': nL
                }

                if j == num_itr - 1:
                    total_loss += L.sum()
                    total_normal_loss += nL.sum()
                    total_size += L.shape[0]

            scenes.append(scene)

        if 'var_dict' in aux_data:
            var_dict = aux_data['var_dict']
            if 'tb_writer' in var_dict and 'global_step' in var_dict:
                writer = var_dict['tb_writer']
                writer.add_scalar('Val sdf loss', total_loss / total_size,
                                  global_step=var_dict['global_step'])
                writer.add_scalar('Val normal loss', total_normal_loss / total_size,
                                  global_step=var_dict['global_step'])
        return scenes
