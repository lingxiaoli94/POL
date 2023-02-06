import torch

def compute_box_distance(box1, box2, share_dim, kind):
    '''
    Args:
        box1: BxNx4, box2: BxMx4 (if share_dim then N==M)
    Returns:
        dist, BxNxM (or BxN if share_dim)
    '''
    if not share_dim:
        box1 = box1.unsqueeze(-2) # BxNxMx4
        box2 = box2.unsqueeze(-3) # BxNxMx4
    diff = box1 - box2
    if kind == 'l1':
        dist = diff.abs().sum(-1)
    elif kind == 'l2':
        dist = diff.square().sum(-1)
    else:
        raise Exception(f'Unknown box loss kind: {kind}')

    return dist

def make_box_distance_fn(kind, center_only=False):
    '''
    kind in ['l1', 'l2']
    '''
    def loss_fn(box1, box2, share_dim):
        if center_only:
            return compute_box_distance(box1[:, :, :2], box2[:, :, :2], share_dim, kind)
        else:
            return compute_box_distance(box1, box2, share_dim, kind)

    return loss_fn
