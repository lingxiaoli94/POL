from .problem_base import ProblemBase
import torch
import numpy as np
from torch.utils.data import TensorDataset

'''
A class of problems where we minimize the energy of a spring system:
o---o---o----o--o

The two endpoints are fixed at (0,0) and (W, 0), and for simplicity we
assume the topology is just a line graph and the spring lives in 2D.

The theta here are the rest lengths of each spring.
When the distance between the two endpoints is less than the sum of rest
lengths, then the spring can bend upwards or downwards, corresponding to
multiple local minima of the total spring energy.
'''
class SpringEquilibriumProblem(ProblemBase):
    def __init__(self,
                 device,
                 num_free,
                 train_thetas,
                 width):
        '''
        num_free: number of free particles, so the total number of particles
        should be (F+2) if including the endpoints, with (F+1) springs.

        train_thetas: Nx(F+1), so that train_thetas[i, :] indicate the rest
        lengths of all (F+1) springs for the ith training problem.

        width: a scalar(=W), representing the distance between the two
        endpoints.
        '''
        self.device = device
        self.num_free = num_free
        self.train_thetas = train_thetas
        self.width = width

    def get_train_dataset(self):
        '''
        In this problem, self.train_thetas has all the information we need,
        so we just wrap it in a TensorDataset.
        '''
        return TensorDataset(self.train_thetas)

    def sample_prior(self, data_batch, batch_size):
        '''
        Return: IxBxD
        In the spring problem, we have D=2F, since each free particle has
        2 coordinates.
        For simplicity, we assume the prior is a uniform distribution over
        [0, W] x [-1, 1], where W=self.width.
        '''
        I = data_batch[0].shape[0]
        B = batch_size
        F = self.num_free
        x = self.width * torch.rand([I, B, F], device=self.device)
        y = 2 * torch.rand([I, B, F], device=self.device) - 1
        samples = torch.stack([x, y], -1) # IxBxFx2
        return samples.reshape(I, B, -1) # IxBx2F

    def unpack_positions(self, X):
        '''
        Unpack X into particle positions as well as adding two endpoints.
        X: IxBx2F
        Return: IxBx(F+2)x2
        '''
        I, B = X.shape[:2]
        # Put in the two endpoints to make code simpler.
        P = X.reshape([X.shape[0], X.shape[1], self.num_free, 2]) # IxBxFx2
        start_point = torch.tensor([0, 0],
                                   device=X.device, dtype=P.dtype)
        end_point = torch.tensor([self.width, 0],
                                 device=X.device, dtype=P.dtype)
        P = torch.cat([
            start_point[None, None, None, :].expand(I, B, -1, -1),
            P,
            end_point[None, None, None, :].expand(I, B, -1, -1)
        ], 2) # IxBx(F+2)x2
        return P

    def eval_loss(self, data_batch, X):
        '''
        X: IxBxD, same format as the output of self.sample_prior
        Return: IxB
        '''
        rest_lengths = data_batch[0] # Ix(F+1)

        positions = self.unpack_positions(X) # IxBx(F+2)x2

        # Calculate spring energy, assuming the spring constants are all 1.
        lengths = torch.sqrt((positions[:, :, :-1, :] - positions[:, :, 1:, :]).square().sum(-1)
                             + 1e-8) # IxBx(F+1)
        energy = (rest_lengths.unsqueeze(1) - lengths).square().sum(-1) # IxB
        return energy

    def extract_latent_code(self, data_batch):
        '''
        In this problem we can directly use theta as the latent code.
        '''
        return data_batch[0]

    def validate(self, solver, aux_data=None):
        '''
        In this function, we decide what to validate (by providing thetas)
        and what to save to the disk.

        In this example, we use a shortcut function simple_validate.
        '''

        from pol.utils.validation.shortcuts import simple_validate

        # See the source code of simple_validate for the description of
        # info_fn and result_fn.
        def info_fn(thetas):
            return {
                'type': 'ndarray_dict',
                'thetas': thetas.detach().cpu()
            }

        def result_fn(thetas, X):
            P = self.unpack_positions(X)
            result = {
                'type': 'ndarray_dict',
                'positions': P.detach().cpu(), # IxBx(F+2)x2
                'loss': self.eval_loss([thetas], X).detach().cpu()
            }
            return result

        return simple_validate(self, solver, aux_data, info_fn, result_fn)
