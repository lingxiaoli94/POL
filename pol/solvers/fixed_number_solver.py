import torch
import numpy as np
import math
from .universal_solver_base import UniversalSolverBase
import pol.problems.objdetect


'''
Special solver for objdetect that predicts a fixed number of boxes.
'''


class FixedNumberSolver(UniversalSolverBase, torch.nn.Module):
    def __init__(self, *,
                 device,
                 latent_dim,
                 max_num_box,
                 apply_sigmoid=False):
        super().__init__()
        self.device = device
        self.max_num_box = max_num_box
        self.apply_sigmoid = apply_sigmoid
        self.box_ffn = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 4 * max_num_box))
        self.box_ffn.to(self.device)

    def set_problem(self, problem):
        super().set_problem(problem)
        assert(isinstance(problem, pol.problems.objdetect.ObjectDetection))

    def predict_boxes(self, data_batch):
        '''
        Return: Ix..., that can be fed to self.pushforward_net.
        '''
        latent_code = self.problem.extract_latent_code(data_batch)
        pred_boxes = self.box_ffn(latent_code) # Ix4M
        if self.apply_sigmoid:
            pred_boxes = pred_boxes.sigmoid()
        return pred_boxes.reshape(-1, self.max_num_box, 4) # IxMx4

    def compute_losses(self, data_batch, prior_samples):
        pred_boxes = self.predict_boxes(data_batch) # IxMx4
        gt_boxes = data_batch['boxes'] # IxKx4
        gt_masks = data_batch['box_masks'] # IxK
        gt_count = gt_masks.float().sum(-1) # I

        M = pred_boxes.shape[1]

        dist = self.problem.box_loss_fn(pred_boxes, gt_boxes, share_dim=False) # IxMxK
        dist_filled = torch.where(gt_masks.unsqueeze(-2).expand(-1, M, -1),
                                  dist,
                                  torch.full_like(dist, 1e8)) # IxBxK
        chamfer_pred_gt = dist_filled.min(-1)[0] # IxB
        chamfer_gt_pred = dist_filled.min(-2)[0] # IxK
        chamfer_loss = chamfer_pred_gt.mean(-1) # I
        chamfer_loss += (torch.where(gt_masks,
                                     chamfer_gt_pred,
                                     torch.zeros_like(chamfer_gt_pred)).sum(-1) /
                         torch.maximum(
                             torch.ones_like(gt_count),
                             gt_count)) # I

        chamfer_loss = torch.where(gt_count > 0,
                                   chamfer_loss,
                                   torch.zeros_like(chamfer_loss))

        loss_dict = {
            'chamfer_loss': chamfer_loss,
        }
        return loss_dict
