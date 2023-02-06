import torch
import numpy as np

def apply_reflection(P, R):
    # P - B1x3 or NxB1x3, R - B2x4 or NxB2x4, assume R is normalized
    # returns: B1xB2x3 or NxB1xB2x3
    normal = R[..., :3]  # NxB2x3
    intercept = R[..., 3]  # NxB2
    dots = (P.unsqueeze(-2) * normal.unsqueeze(-3)).sum(-1)  # NxB1xB2
    sd = dots - intercept.unsqueeze(-2)  # NxB1xB2

    P_ref = P.unsqueeze(-2) - 2 * sd.unsqueeze(-1) * normal.unsqueeze(-3)  # NxB1xB2x3
    return P_ref


def normalize_reflection(Z):
    # Z - Bx4 or NxBx4
    # returns: Bx4 or NxBx4, normal+intercept
    normal = Z[..., :3]
    normal = normal / (normal.norm(dim=-1, keepdim=True) + 1e-6)
    intercept = Z[..., -1]
    return torch.cat([normal, intercept.unsqueeze(-1)], -1)


def calc_verification_loss(R, P, sampler, index_batch,
                           fidelity_exp_theta, mode='mean',
                           single_shape=False, l1_norm=False,
                           lse_exp_theta=100.0):
    # P - NxB1x3
    # R - NxB2x4, normal+intercept, returns: NxB2
    # Compute verification loss.
    def sdf_fn(P):
        if single_shape:
            assert(index_batch.shape[0] == 1)
            return sampler.compute_sdf(
                index_batch, P.reshape(1, -1, 3)).reshape(P.shape[:2])
        else:
            return sampler.compute_sdf(index_batch, P)

    P_sdf = sdf_fn(P)  # NxB1
    weights = torch.exp(fidelity_exp_theta * -P_sdf.abs())  # NxB1
    P_ref = apply_reflection(P, R)  # NxB1xB2x3
    P_ref_sdf = sdf_fn(
        P_ref.reshape(P.shape[0], -1, 3)
    ).reshape(P_ref.shape[:3])  # NxB1xB2

    if l1_norm:
        sdf_diff = (P_sdf.unsqueeze(-1) - P_ref_sdf).abs()
    else:
        sdf_diff = (P_sdf.unsqueeze(-1) - P_ref_sdf).square()
    sdf_diff = (
        sdf_diff *
        weights.unsqueeze(-1))  # NxB1xB2

    if mode == 'full':
        return sdf_diff # NxB1xB2

    if mode == 'lse':
        # This does not seem to help.
        sdf_diff = (torch.logsumexp(lse_exp_theta * sdf_diff, 1) -
                    torch.log(torch.tensor(sdf_diff.shape[1])).to(sdf_diff))  # NxB2
    elif mode == 'neg_lse':
        # Prioritize closely matched patches.
        sdf_diff = -(torch.logsumexp(-lse_exp_theta * sdf_diff, 1)).to(sdf_diff)  # NxB2
    elif mode == 'mean':
        sdf_diff = sdf_diff.mean(1)  # NxB2
    else:
        assert(mode == 'max')
        sdf_diff = sdf_diff.max(1)[0]  # NxB2
    return sdf_diff

