import torch
import meshio
import geomlib
import numpy as np
import h5py
import argparse
from pol.problems.basic_samplers import UniformSampler


class OfflineMeshSampler:
    def __init__(self,
                 mesh_file,
                 out_file,
                 num_surface_samples,
                 num_ambient_samples,
                 surface_jitter=0.0,
                 sdf_origin=[0, 0, 0]):
        device = torch.device('cuda')
        mesh = meshio.read(mesh_file)
        vertices = mesh.points
        self.vertices = vertices

        faces = mesh.cells_dict['triangle']
        self.faces = faces

        vertices_d = torch.from_numpy(vertices).to(device)
        faces_d = torch.from_numpy(faces).to(device)
        self.info_d = geomlib.TriangularProjectionInfo(vertices_d, faces_d)
        self.origin = torch.tensor(sdf_origin).to(device).to(torch.float32)
        self.surface_sampler_d = geomlib.TriangularMeshSampler(vertices_d, faces_d)
        self.ambient_sampler_d = UniformSampler(
            device,
            low=[-1, -1, -1],
            high=[1, 1, 1])
        self.out_file = out_file
        self.num_surface_samples = num_surface_samples
        self.num_ambient_samples = num_ambient_samples
        self.surface_jitter = surface_jitter

    def run(self):
        surface_samples = self.surface_sampler_d.sample(
            self.num_surface_samples)
        if self.surface_jitter > 0:
            noise = self.surface_jitter * torch.randn(surface_samples.shape).to(surface_samples)
            surface_samples += noise

        ambient_samples = self.ambient_sampler_d.sample(self.num_ambient_samples)
        all_samples = torch.cat([surface_samples, ambient_samples], 0)

        sdfs = geomlib.signed_distance(all_samples, self.origin, self.info_d)

        # Bring everything back to CPU.
        all_samples = all_samples.detach().cpu().numpy()
        sdfs = sdfs.detach().cpu().numpy()

        print('Start writing to h5 file...')
        fh = h5py.File(self.out_file, 'w')
        fh.create_dataset('samples', data=all_samples,
                          compression='gzip', compression_opts=9)
        fh.create_dataset('sdfs', data=sdfs,
                          compression='gzip', compression_opts=9)
        fh.create_dataset('vertices', data=self.vertices,
                          compression='gzip', compression_opts=9)
        fh.create_dataset('faces', data=self.faces,
                          compression='gzip', compression_opts=9)
        print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mesh_file', type=str)
    parser.add_argument('out_file', type=str)
    parser.add_argument('num_surface_samples', type=int)
    parser.add_argument('num_ambient_samples', type=int)
    parser.add_argument('surface_jitter', type=float)
    args = parser.parse_args()

    sampler = OfflineMeshSampler(
        mesh_file=args.mesh_file,
        out_file=args.out_file,
        num_surface_samples=args.num_surface_samples,
        num_ambient_samples=args.num_ambient_samples,
        surface_jitter=args.surface_jitter)
    sampler.run()
