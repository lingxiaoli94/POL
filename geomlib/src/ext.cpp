#include <torch/extension.h>

#include "generalized_projection.h"
#include "generalized_projection_info.h"
#include "point_tet_mesh_test.h"
#include "point_tri_mesh_test.h"
#include "signed_distance.h"
#include "triangular_mesh_sampler.h"

namespace py = pybind11;
using namespace geomlib;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<TriangularProjectionInfo>(m, "TriangularProjectionInfo")
      .def(py::init<torch::Tensor, torch::Tensor>())
      .def_readwrite("vertices", &TriangularProjectionInfo::vertices)
      .def_readwrite("faces", &TriangularProjectionInfo::faces)
      .def_readwrite("v0", &TriangularProjectionInfo::v0)
      .def_readwrite("e1", &TriangularProjectionInfo::e1)
      .def_readwrite("e2", &TriangularProjectionInfo::e2)
      .def("bary_points",
           &TriangularProjectionInfo::GetPointsAtBarycentricCoordinates);

  py::class_<TriangularMeshSampler>(m, "TriangularMeshSampler")
      .def(py::init<torch::Tensor, torch::Tensor>())
      .def("sample", &TriangularMeshSampler::Sample)
      .def("sample_with_weights", &TriangularMeshSampler::SampleWithWeights);

  m.def("triangle_projection_3d", &ComputeGeneralizedTriangleProjection<3>);

  m.def("point_tri_mesh_test", &PointTriMeshTest);
  m.def("point_tet_mesh_test", &PointTetMeshTest);
  m.def("signed_distance", &PointTriMeshSignedDistance);
};
