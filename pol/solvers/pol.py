import torch
import numpy as np
import math
from .universal_solver_base import UniversalSolverBase


'''
Proximal operator learning.
'''


class POL(UniversalSolverBase, torch.nn.Module):
    def __init__(self,
                 device, *,
                 network_formula,
                 num_horizon,
                 discount_factor=1.0):
        super().__init__()
        self.device = device
        self.num_horizon = num_horizon
        self.pushforward_net = network_formula.create_instance()
        self.pushforward_net.to(self.device)
        self.discount_factor = discount_factor

    def set_problem(self, problem):
        super().set_problem(problem)

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

        problem_loss = 0
        transport_loss = 0

        Z_cur = prior_samples
        cur_discount = 1.0
        for i in range(self.num_horizon):
            Z_prev = Z_cur.detach()
            Z_next = self.push_forward(data_batch, latent_code, Z_prev)  # IxBxD
            transport_loss += cur_discount * self.problem.compute_distance(
                data_batch, Z_next, Z_prev) # IxB
            if self.problem.has_ground_truth():
                problem_loss += cur_discount * self.problem.eval_loss(
                    data_batch, Z_next, Z_prev
                ) # IxB
            else:
                problem_loss += cur_discount * self.problem.eval_loss(
                    data_batch, Z_next)  # IxB
            cur_discount *= self.discount_factor
            Z_cur = Z_next

        loss_dict = {
            'problem_loss': problem_loss,
            'transport_loss': transport_loss
        }
        return loss_dict
