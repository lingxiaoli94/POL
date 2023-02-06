from pybind11.setup_helpers import ParallelCompile
from glob import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = [
    CUDAExtension('geomlib', [
        'src/ext.cpp',
        'src/generalized_projection_info.cpp',
        'src/generalized_projection_cuda.cu',
        'src/point_tri_mesh_test_cuda.cu',
        'src/point_tet_mesh_test_cuda.cu',
        'src/signed_distance.cpp',
        'src/triangular_mesh_sampler.cpp',
    ])
]

ParallelCompile("NPY_NUM_BUILD_JOBS").install()

setup(
    name='geomlib',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    }
)
