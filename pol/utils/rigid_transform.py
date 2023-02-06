import torch
import numpy as np


def apply_rigid_transform(P, T):
    '''
    P: B1x3 or NxB1x3
    T: B2x6 or NxB2x6
    Return: B1xB2x3 or NxB1xB2x3
    '''
    B1 = P.shape[-2]
    B2 = T.shape[-2]
    axis = T[..., :3]
    translation = T[..., 3:]

    angle = axis.norm(dim=-1, keepdim=True)  # NxB2x1
    k = axis / angle  # NxB2x3

    P_ex = P.unsqueeze(-2)  # NxB1x1x3
    translation_ex = translation.unsqueeze(-3)  # Nx1xB2x3
    angle_ex = angle.unsqueeze(-3)  # Nx1xB2x1
    k_ex = k.unsqueeze(-3)  # Nx1xB2x3
    cross_prod = torch.cross(k_ex.expand(-1, B1, -1, -1),
                             P_ex.expand(-1, -1, B2, -1), -1)  # NxB1xB2x3
    dot_prod = (P_ex * k_ex).sum(-1, keepdim=True)  # NxB1xB2x1
    cos_angle_ex = torch.cos(angle_ex)
    # Rodrigues rotation formula.
    P_rot = (P_ex * cos_angle_ex + cross_prod * torch.sin(angle_ex) +
             k_ex * dot_prod * (1 - cos_angle_ex))  # NxB1xB2x3
    return P_rot + translation_ex
