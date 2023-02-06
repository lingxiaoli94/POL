import torch
import numpy as np
from .problem_base import ProblemBase
from torch.utils.data import TensorDataset
from pol.utils.batch_eval import batch_eval, batch_eval_index


class AnalyticalProblem(ProblemBase):
    def __init__(self,
                 device,
                 var_dim,
                 fn,
                 train_thetas,
                 var_bbox):
        '''
        var_dim: dimension of the search sapce, (=D)
        fn: (X, theta) -> Y, where
            X: ...xD
            theta: ...xC
            Y: ...
        It is assumed that there is no stochasticity in fn.
        train_thetas: NxC
        var_bbox: Dx2, describes the bbox of the search space
        '''
        self.device = device
        self.var_dim = var_dim
        self.fn = fn
        self.train_thetas = train_thetas  # NxC
        self.var_bbox = var_bbox.to(device)  # Dx2

    def get_train_dataset(self):
        return TensorDataset(self.train_thetas)

    def sample_prior(self, data_batch, batch_size):
        '''
        data_batch: a list of a singleton theta.
        Return: IxBxD
        '''
        X = torch.rand([data_batch[0].shape[0], batch_size, self.var_dim],
                       device=self.device)  # IxBxD
        return (X * (self.var_bbox[:, 1] - self.var_bbox[:, 0]) +
                self.var_bbox[:, 0])

    def do_satisfy(self, data_batch, X):
        '''
        X: IxBxD
        Return: IxB
        '''
        mask = torch.logical_and(X >= self.var_bbox[:, 0],
                                 X <= self.var_bbox[:, 1]) # IxBxD
        return mask.all(-1)

    def eval_loss(self, data_batch, X):
        '''
        X: IxBxD
        Return: IxB
        '''
        thetas = data_batch[0]  # IxC
        thetas = thetas.unsqueeze(1).expand(-1, X.shape[1], -1)  # IxBxC
        Y = self.fn(X, thetas)  # IxB
        return Y

    def extract_latent_code(self, data_batch):
        '''
        Return: IxC
        '''
        return data_batch[0]

    def validate(self, solver, aux_data):
        '''
        aux_data should contain:
            * test_thetas
            * (optional) include_loss
            * (optional) include_witness
            * (optional) withness_batch_size
            * (optional) loss_landscape, a dictionary containing:
                - points, WxHx2, on which we evaluate the loss
            * (optional) gt_prox_fn, a function that takes IxBxD -> scalar
        '''
        from pol.utils.validation.shortcuts import simple_validate

        def info_fn(thetas):
            result = {
                'type': 'ndarray_dict',
                'thetas': thetas.detach().cpu()
            }
            if aux_data.get('include_witness', False):
                for i in range(10):
                    witness = self.sample_prior([thetas],
                                                aux_data.get('witness_batch_size', 1024))
                    result[f'witness_{i}'] = witness.detach().cpu()

            landscape = aux_data.get('loss_landscape', None)
            if landscape is not None:
                from pol.utils.validation.shortcuts import eval_landscape
                result.update(eval_landscape(thetas=thetas,
                                             landscape=landscape,
                                             eval_fn=lambda D, X: self.eval_loss(D, X)))
            if aux_data.get('gt_prox_fn', None):
                # This is just to verify proximal operator is correct in L2 sense.
                gt_prox_fn = aux_data['gt_prox_fn']
                prior_samples = self.sample_prior([thetas],
                                                          aux_data.get('prior_batch_size', 1024))
                X_detach = prior_samples.detach()
                gt = gt_prox_fn(X_detach)
                batch_size = aux_data.get('instance_batch_size', 32)
                X_cpu = batch_eval_index(
                    lambda inds: solver.extract_pushed_samples([thetas[inds]], X_detach[inds]),
                    thetas.shape[0],
                    dim=0,
                    batch_size=batch_size,
                    no_tqdm=True)
                X = X_cpu.to(solver.device)
                mse = (X - gt).square().sum(-1).mean()
                mse = mse / X.shape[-1] # downscale by dimension
                if 'var_dict' in aux_data:
                    var_dict = aux_data['var_dict']
                    if 'tb_writer' in var_dict and 'global_step' in var_dict:
                        writer = var_dict['tb_writer']
                        writer.add_scalar('gt MSE', mse,
                                          global_step=var_dict['global_step'])


            return result

        def result_fn(thetas, X):
            result = {
                'type': 'ndarray_dict',
                'X': X.detach().cpu()
            }
            if aux_data.get('include_loss', True):
                result['loss'] = self.eval_loss([thetas], X).detach().cpu()
                result['satisfy'] = self.do_satisfy([thetas], X).detach().cpu()

            return result

        return simple_validate(self, solver, aux_data, info_fn, result_fn)
