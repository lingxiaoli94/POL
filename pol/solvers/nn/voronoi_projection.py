import torch

class VoronoiProjection(torch.nn.Module):
    def __init__(self,
                 distance_fn):
        super().__init__()
        self.distance_fn = distance_fn

    def forward(self, F, x):
        '''
        Args:
            F: IxKxD,
            x: IxBxD
        Returns:
            proj_x: IxBxD

        Simply project x[i, b, :] to the closest F[i, k, :].
        '''
        assert(F.shape[-1] == x.shape[-1] and F.dim() == 3)
        dist = self.distance_fn(x, F, share_dim=False) # IxBxK
        _, closest_idx = dist.min(-1) # IxB
        closest_idx = closest_idx.unsqueeze(-1).repeat(1, 1, x.shape[-1]) # IxBxD
        return torch.gather(F, 1, closest_idx)
