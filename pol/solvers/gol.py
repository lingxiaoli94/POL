import torch
import numpy as np
import math
from pol.utils.jacobian import compute_jacobian
from .universal_solver_base import UniversalSolverBase

'''
Gradient operator learning.
'''
class GOL(UniversalSolverBase, torch.nn.Module):
    def __init__(self,
                 device, *,
                 network_formula,
                 num_grad_step,
                 num_horizon,
                 lr):
        super().__init__()
        self.device = device
        self.num_grad_step = num_grad_step
        self.num_horizon = num_horizon
        self.lr = lr
        self.pushforward_net = network_formula.create_instance()
        self.pushforward_net.to(self.device)

    def compute_latent_code(self, data_batch):
        '''
        Return: Ix..., that can be fed to self.pushforward_net.
        '''
        latent_code = self.problem.extract_latent_code(data_batch)
        return latent_code

    def push_forward(self, data_batch, latent_code, X):
        '''
        Save time by reusing latent_code.
        '''
        X = self.problem.apply_pre_push(data_batch, X)
        X_pushed = self.problem.push_forward(
            latent_code, X, self.pushforward_net)
        return self.problem.apply_projection(data_batch, X_pushed)

    def extract_pushed_samples(self, data_batch, prior_samples):
        '''
        prior_samples: IxBxD
        '''
        latent_code = self.compute_latent_code(data_batch) # IxC
        return self.push_forward(data_batch, latent_code, prior_samples)

    def compute_losses(self, data_batch, prior_samples):
        '''
        prior_sampels: IxBxD
        '''
        latent_code = self.compute_latent_code(data_batch) # IxC
        regression_loss = 0

        X_prev = prior_samples
        for i in range(self.num_horizon):
            X = X_prev.detach() # IxBxD
            for k in range(self.num_grad_step):
                # Perform gradient descent on prior_samples.
                X_detach = X.detach()
                X_detach.requires_grad_(True)
                if self.problem.has_ground_truth():
                    Y = self.problem.eval_loss(data_batch, X_detach, X_prev) # IxB
                else:
                    Y = self.problem.eval_loss(data_batch, X_detach) # IxB
                J = compute_jacobian(Y.unsqueeze(-1), X_detach, create_graph=False, retain_graph=False) # IxBx1xD
                J = J.squeeze(-2).detach() # IxBxD, gradient

                X = self.problem.apply_projection(data_batch, X - self.lr * J).detach() # IxBxD
            X_pushed = self.push_forward(data_batch, latent_code, X_prev) # IxBxD
            regression_loss += self.problem.compute_distance(
                data_batch, X, X_pushed) # IxB
            X_prev = X_pushed.detach()

        return {
            'regression_loss': regression_loss,
        }
