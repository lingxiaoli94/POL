import torch
import numpy as np
from pol.utils.tri_mesh import TriMesh

'''
A dataset-like class that supports lazy-loading of meshes.
'''
class MeshRepository:
    def __init__(self,
                 mesh_file_list,
                 *,
                 scale_to_unit_cube=True,
                 fixed_pc_count,
                 resample_fixed_pc=True):
        self.num_mesh = len(mesh_file_list)
        self.mesh_file_list = mesh_file_list
        self.scale_to_unit_cube = scale_to_unit_cube
        self.fixed_pc_count = fixed_pc_count
        self.resample_fixed_pc = resample_fixed_pc

        self.mesh_cache = {}
        self.fixed_pc_cache = {}
        self.fixed_pc_with_normal_cache = {}

    def get_num_mesh(self):
        return self.num_mesh

    def get_mesh_lazy(self, i):
        assert(i < self.num_mesh)
        if i not in self.mesh_cache:
            self.mesh_cache[i] = TriMesh.from_file(
                self.mesh_file_list[i],
                scale_to_unit_cube=self.scale_to_unit_cube
            )
        return self.mesh_cache[i]

    def get_fixed_pc_lazy(self, i):
        assert(i < self.num_mesh)
        if (self.resample_fixed_pc) or (i not in self.fixed_pc_cache):
            mesh = self.get_mesh_lazy(i)
            self.fixed_pc_cache[i] = mesh.sample_surface(
                self.fixed_pc_count, include_normal=False).cpu()
        return self.fixed_pc_cache[i]

    def get_fixed_pc_with_normal_lazy(self, i):
        assert(i < self.num_mesh)
        if (self.resample_fixed_pc) or (i not in self.fixed_pc_with_normal_cache):
            mesh = self.get_mesh_lazy(i)
            self.fixed_pc_with_normal_cache[i] = mesh.sample_surface(
                self.fixed_pc_count, include_normal=True).cpu()
        return self.fixed_pc_with_normal_cache[i]

    def get_fixed_pcs(self, index_batch, include_normal):
        pcs = []
        for idx in index_batch:
            if include_normal:
                pc = self.get_fixed_pc_with_normal_lazy(idx.item())
            else:
                pc = self.get_fixed_pc_lazy(idx.item())
            pcs.append(pc)
        return torch.stack(pcs, 0).cuda()

    def get_normals(self, index_batch, face_idxs):
        '''
        Args:
            face_idxs: IxB
        Return:
            normals, IxBx3
        '''
        all_normals = []
        for i, mesh_idx in enumerate(index_batch):
            mesh = self.get_mesh_lazy(mesh_idx.item())
            face_idx_i = face_idxs[i, :]
            normals = mesh.get_normals_cuda()[face_idx_i, :]
            all_normals.append(normals)
        return torch.stack(all_normals, 0)

    def compute_sdf(self, index_batch, points):
        '''
        Args:
            points: IxBx3
        Return:
            (signed) L2 distance (after sqrt), IxB
            closest face index, IxB
        '''
        all_sdfs = []
        all_inds = []
        for i, mesh_idx in enumerate(index_batch):
            mesh = self.get_mesh_lazy(mesh_idx.item())
            sdfs, inds = mesh.compute_sdf_gpu(points[i, :, :].cuda())
            all_sdfs.append(sdfs)
            all_inds.append(inds)
        all_sdfs = torch.stack(all_sdfs, 0).to(points) # IxB
        all_inds = torch.stack(all_inds, 0).to(points.device) # IxB

        return all_sdfs, all_inds

