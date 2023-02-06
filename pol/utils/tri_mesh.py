import torch
import numpy as np
import meshio
import geomlib

from pol.utils.scaling import scale_to_unit_cube as scale_to_cube
from pol.utils.sdf import SignedDistanceFunction

class TriMesh:
    def __init__(self, V, F, *,
                 transform=None,
                 scale_to_unit_cube=None,
                 dtype=torch.float32,
                 cache_info=False):
        if scale_to_unit_cube:
            V = scale_to_cube(V)
        if transform is not None:
            V = np.matmul(V, transform.T)

        self.vertices = torch.from_numpy(V).to(dtype)
        self.faces = torch.from_numpy(F)
        self.compute_normals()
        self.dtype = torch.float32
        self.cache_info = cache_info
        self.cached_info_d = None

    def compute_normals(self):
        vertices = self.vertices
        faces = self.faces
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        normals = (v1 - v0).cross(v2 - v0)
        normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
        self.normals = normals

    @staticmethod
    def from_file(f, **kwargs):
        mesh = meshio.read(f)
        return TriMesh(mesh.points, mesh.cells_dict['triangle'], **kwargs)

    def get_vertices_cuda(self):
        if self.cache_info and self.cached_info_d is not None:
            return self.cached_info_d.vertices
        return self.vertices.cuda()

    def get_faces_cuda(self):
        if self.cache_info and self.cached_info_d is not None:
            return self.cached_info_d.faces
        return self.faces.cuda()

    def get_normals_cuda(self):
        return self.normals.cuda()

    def sample_surface(self, batch_size, *,
                       include_normal=False,
                       generator=None):
        '''
        Return:
            If include_normal, then Bx6. Otherwise Bx3.
        '''
        vertices_d = self.get_vertices_cuda()
        faces_d = self.get_faces_cuda()

        v0 = vertices_d[faces_d[:, 0]]
        v1 = vertices_d[faces_d[:, 1]]
        v2 = vertices_d[faces_d[:, 2]]
        area_vec = (v1 - v0).cross(v2 - v0)
        area = (area_vec.square().sum(-1) + 1e-8).sqrt() / 2

        chosen_idxs = torch.multinomial(area + 1e-8, batch_size,
                                        replacement=True,
                                        generator=generator)
        v0 = vertices_d[faces_d[chosen_idxs, 0]]
        v1 = vertices_d[faces_d[chosen_idxs, 1]]
        v2 = vertices_d[faces_d[chosen_idxs, 2]]
        vs = torch.stack([v0, v1, v2], 2) # Bx3x3

        r1 = torch.rand([batch_size], device=vertices_d.device,
                        dtype=self.dtype, generator=generator)
        r2 = torch.rand([batch_size], device=vertices_d.device,
                        dtype=self.dtype, generator=generator)
        r1_sqrt = (r1 + 1e-10).sqrt()
        weights = torch.stack([
            1 - r1_sqrt,
            r1_sqrt * (1 - r2),
            r2 * r1_sqrt
        ], 1) # Bx3
        points = (weights.unsqueeze(1) * vs).sum(-1)
        if include_normal:
            normals_d = self.get_normals_cuda()
            normals = normals_d[chosen_idxs]
            return torch.cat([points, normals], -1) # Bx6
        else:
            return points

    def get_info_cuda(self):
        if self.cache_info and self.cached_info_d is not None:
            return self.cached_info_d
        else:
            vertices_d = self.get_vertices_cuda()
            faces_d = self.get_faces_cuda()
            info_d = geomlib.TriangularProjectionInfo(vertices_d, faces_d)

            if self.cache_info:
                self.cached_info_d = info_d
            return info_d

    def compute_sdf_gpu(self, points):
        assert(points.is_cuda)
        info_d = self.get_info_cuda()
        return SignedDistanceFunction.apply(
            points.to(self.dtype),
            info_d, True)
