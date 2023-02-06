import torch
import meshio
import geomlib
import numpy as np
from pol.utils.sdf import SignedDistanceFunction
from pol.utils.scaling import scale_to_unit_cube
from pol.problems.basic_samplers import UniformSampler


class MeshProjector:
    def __init__(self, *,
                 device,
                 vertices=None,
                 faces=None,
                 mesh_file=None,
                 scaling=False,
                 transform=None,
                 surface_ratio=1.0,
                 surface_jitter=0.0,
                 sdf_origin=[0, 0, 0],
                 dtype=torch.float32,
                 unsigned=True):
        assert(device.type == 'cuda')
        if mesh_file is not None:
            mesh = meshio.read(mesh_file)
            faces = mesh.cells_dict['triangle']
            vertices = mesh.points
        else:
            assert(vertices is not None and faces is not None)
        if scaling:
            vertices = scale_to_unit_cube(vertices)
        if transform is not None:
            vertices = np.matmul(vertices, transform.T)
        self.vertices = vertices
        self.faces = faces

        vertices_d = torch.from_numpy(vertices).to(device).to(dtype)
        faces_d = torch.from_numpy(faces).to(device)

        self.info_d = geomlib.TriangularProjectionInfo(vertices_d, faces_d)
        self.origin_d = torch.tensor(sdf_origin).to(device).to(dtype)
        self.surface_sampler_d = geomlib.TriangularMeshSampler(vertices_d, faces_d)
        self.ambient_sampler_d = UniformSampler(
            device,
            low=[-1, -1, -1],
            high=[1, 1, 1])
        self.surface_ratio = surface_ratio
        self.surface_jitter = surface_jitter
        self.unsigned = unsigned
        self.dtype = dtype

    def sample_ext(self,
                   surface_batch_size,
                   ambient_batch_size,
                   surface_jitter):
        if surface_batch_size > 0:
            surface_samples = self.surface_sampler_d.sample(
                surface_batch_size)
            if surface_jitter > 0:
                noise = surface_jitter * torch.randn(
                    surface_samples.shape).to(surface_samples)
                surface_samples += noise
        else:
            surface_samples = None

        if ambient_batch_size > 0:
            ambient_samples = self.ambient_sampler_d.sample(
                ambient_batch_size)
        else:
            ambient_samples = None

        all_samples = []
        if surface_samples is not None:
            all_samples.append(surface_samples)
        if ambient_samples is not None:
            all_samples.append(ambient_samples)
        all_samples = torch.cat(all_samples, 0)

        return all_samples.to(self.dtype)

    def sample(self, batch_size):
        num_surface_sample = int(self.surface_ratio * batch_size)
        return self.sample_ext(
            num_surface_sample,
            batch_size - num_surface_sample,
            self.surface_jitter
        )

    def compute_sdf(self, points):
        return SignedDistanceFunction.apply(
            points.to(self.dtype),
            self.info_d, self.unsigned, self.origin_d).to(torch.float32)
