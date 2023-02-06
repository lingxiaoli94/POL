import numpy as np
import torch
import scipy
import igl


class WaveKernelSignature:
    def __init__(self,
                 vertices,
                 faces,
                 top_k_eig=200,
                 timestamps=100):
        # vertices, faces are both numpy arrays.
        self.vertices = vertices
        self.faces = faces

        self.vertices_gpu = torch.from_numpy(vertices).cuda()
        self.faces_gpu = torch.from_numpy(faces).cuda()

        self.top_k_eig = top_k_eig
        self.timestamps = timestamps

    def prepare(self):
        # Prepare everything on CPU.
        L = -igl.cotmatrix(self.vertices, self.faces)
        M = igl.massmatrix(self.vertices, self.faces, igl.MASSMATRIX_TYPE_VORONOI)

        self.eig_vals, self.eig_vecs = scipy.sparse.linalg.eigsh(
            L, self.top_k_eig, M, sigma=0, which='LM')
        self.eig_vecs /= np.linalg.norm(self.eig_vecs, axis=0, keepdims=True)

        delta = (np.log(self.eig_vals[-1]) - np.log(self.eig_vals[1])) / self.timestamps
        sigma = 7 * delta
        e_min = np.log(self.eig_vals[1]) + 2 * delta
        e_max = np.log(self.eig_vals[-1]) - 2 * delta
        print('wks_e_min: {}, wks_e_max: {}'.format(e_min, e_max))
        es = np.linspace(e_min, e_max, self.timestamps)  # T
        self.delta = delta

        coef = np.expand_dims(es, 0) - np.expand_dims(np.log(self.eig_vals[1:]), 1)  # (K-1)xT
        coef = np.exp(-np.square(coef) / (2 * sigma * sigma))  # (K-1)xT
        sum_coef = coef.sum(0)  # T
        K = np.matmul(np.square(self.eig_vecs[:, 1:]),  coef)  # VxT
        self.wks = K / np.expand_dims(sum_coef, 0)  # VxT
        self.wks_gpu = torch.from_numpy(self.wks).cuda()

    def calc_wks_dist(self, batch_from):
        # batch_from is an index array on GPU.
        dist_sub = self.wks_gpu[batch_from, :].unsqueeze(1) - self.wks_gpu.unsqueeze(0)  # BxVxT
        dist_plus = self.wks_gpu[batch_from, :].unsqueeze(1) + self.wks_gpu.unsqueeze(0)  # BxVxT
        return (dist_sub / dist_plus).abs().sum(-1) * self.delta  # BxV

    def generate_correspondence(self,
                                batch_size,
                                min_dist=0.1,
                                top_k=8):
        # Returns Cx2x3

        g_gpu = torch.Generator('cuda')
        g_gpu.manual_seed(42)

        perm = torch.randperm(self.vertices_gpu.shape[0], generator=g_gpu, device=self.vertices_gpu.device)
        indices = perm[:batch_size]  # C
        # indices = perm[[1, 3, 5, 6, 9, 13]]
        d_wks = self.calc_wks_dist(indices)  # CxV
        d_euc = (self.vertices_gpu[indices, :].unsqueeze(1) -
                 self.vertices_gpu.unsqueeze(0)).norm(dim=-1)  # CxV
        d_wks = torch.where(d_euc < min_dist, torch.full_like(d_wks, 1e100), d_wks)
        top_wks, top_indices = torch.topk(d_wks, top_k, dim=-1, largest=False)  # CxK
        mask = torch.nn.functional.one_hot(
            torch.randint(high=top_k, size=indices.shape, generator=g_gpu, device='cuda'),
            num_classes=top_k)
        top_indices = (mask * top_indices).sum(-1)  # C
        top_wks = (mask * top_wks).sum(-1) # C
        return (torch.stack([self.vertices_gpu[indices, :],
                             self.vertices_gpu[top_indices, :]], 1),  # Cx2x3
                indices,
                top_indices,
                top_wks)
