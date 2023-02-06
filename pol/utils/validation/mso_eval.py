import torch
import numpy as np
from tqdm import tqdm

def hausdorff_distance(P, Q):
    '''
    Args:
        P: NxD
        Q: MxD
    '''
    assert(P.shape[1] == Q.shape[1])
    N = P.shape[0]
    M = Q.shape[0]
    if N == 0 and M == 0:
        return 0
    if N == 0 or M == 0:
        return np.inf
    C = (P.unsqueeze(1) - Q.unsqueeze(0)).square().sum(-1) # NxM
    C = C.sqrt()

    return max(C.min(1)[0].max(), C.min(0)[0].max())

def distance_sqr_matrix(P, Q):
    '''
    Args:
        P: NxD
        Q: MxD
    '''
    assert(P.shape[1] == Q.shape[1])
    N = P.shape[0]
    M = Q.shape[0]
    assert(N > 0 and M > 0)

    # TODO: batch_eval
    C = (P.unsqueeze(1) - Q.unsqueeze(0)).square().sum(-1) # NxM
    return C

class MSOSolution:
    def __init__(self, X, F, S):
        '''
        Args:
            X: IxBxD
            F: IxB, objective values
            S: IxB, satisfy mask
        '''
        assert(X.ndim == 3 and F.ndim == 2 and S.ndim == 2)
        assert(X.shape[0] == F.shape[0] == S.shape[0])
        assert(X.shape[1] == F.shape[1] == S.shape[1])
        self.X = X
        self.F = F
        self.S = S

    def get_num_instance(self):
        return self.X.shape[0]

    def get(self, i):
        return (self.X[i], self.F[i], self.S[i])

    def get_batch_size(self):
        return self.X.shape[1]

class MSOEvaluation:
    def __init__(self,
                 witness, *,
                 inf_dist,
                 obj_threshold,
                 relative_obj=False,
                 precision_thresholds,
                 use_double_proj=True,
                 gt_solution,
                 candidate_solution,
                 cuda=False,
                 quiet=False):
        '''
        All tensors are torch tensors.
        '''
        self.witness = witness
        self.inf_dist = inf_dist
        self.obj_threshold = obj_threshold
        self.relative_obj = relative_obj
        self.precision_thresholds = precision_thresholds
        self.use_double_proj = use_double_proj
        self.gt_sol = gt_solution
        self.candidate_sol = candidate_solution
        self.cuda = cuda
        self.quiet = quiet


    def witness_distance(self, i, P, Q):
        '''
        Args:
            i: use self.witness[i, :, :]
        '''
        assert(P.shape[1] == Q.shape[1])
        N = P.shape[0]
        M = Q.shape[0]
        if N == 0 and M == 0:
            return 0.0
        if N == 0 or M == 0:
            return self.inf_dist
        witness = self.witness[i, :, :]
        if self.cuda:
            witness = witness.cuda()
            P = P.cuda()
            Q = Q.cuda()
        C_p = distance_sqr_matrix(witness, P) # BxN
        C_q = distance_sqr_matrix(witness, Q) # BxM

        i_p = C_p.min(1)[1] # B
        i_q = C_q.min(1)[1] # B

        if not self.use_double_proj:
            dist = (P[i_p, :] - Q[i_q, :]).square().sum(-1).sqrt() # B
        else:
            C_pq = distance_sqr_matrix(P, Q) # NxM
            dist = (C_pq[i_p, :].min(1)[0].sqrt() +
                    C_pq[:, i_q].min(0)[0].sqrt()) / 2

        if self.cuda:
            dist = dist.cpu()
        return dist # B

    def eval(self):
        self.eval_results = []
        num_instance = self.gt_sol.get_num_instance()
        assert(num_instance == self.candidate_sol.get_num_instance())
        gt_batch_size = self.gt_sol.get_batch_size()
        candidate_batch_size = self.candidate_sol.get_batch_size()
        pbar = range(num_instance) if self.quiet else tqdm(range(num_instance))
        for i in pbar:
            gt_X, gt_F, gt_S = self.gt_sol.get(i)
            X, F, S = self.candidate_sol.get(i)

            if self.relative_obj:
                obj_threshold = gt_F[gt_S].min() * self.obj_threshold
            else:
                obj_threshold = self.obj_threshold
            Xk = X[torch.logical_and(S, F <= obj_threshold), :]
            gt_Xk = gt_X[torch.logical_and(gt_S, gt_F <= obj_threshold), :]

            d_witness = self.witness_distance(i, Xk, gt_Xk)
            precision_thresholds = self.precision_thresholds
            if isinstance(d_witness, float):
                divergence = d_witness
                precision = [(1.0 if d_witness == 0 else 0.0)
                             for threshold in precision_thresholds]
            else:
                divergence = d_witness.mean()
                precision = [(d_witness < threshold).sum() / d_witness.shape[0]
                             for threshold in precision_thresholds]

            self.eval_results.append({
                'divergence': divergence,
                'precision': torch.as_tensor(precision),
                'gt_count': gt_Xk.shape[0] / gt_X.shape[0],
                'candidate_count': Xk.shape[0] / X.shape[0]
            })


        self.avg_results = {}
        first_result = self.eval_results[0]
        for key in first_result:
            if key == 'precision':
                self.avg_results[key] = torch.stack([r[key] for r in self.eval_results],
                                                   -1).mean(-1)
            else:
                self.avg_results[key] = torch.tensor([r[key] for r in self.eval_results],
                                                    dtype=torch.float32).mean()

