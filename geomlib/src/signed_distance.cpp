#include "signed_distance.h"

#include "generalized_projection.h"
#include "point_tri_mesh_test.h"

namespace geomlib {
torch::Tensor PointTriMeshSignedDistance(torch::Tensor points,
                                         torch::Tensor origin,
                                         const TriangularProjectionInfo& info) {
  auto dists = ComputeGeneralizedTriangleProjection<3>(points, info)[0];
  auto signs = PointTriMeshTest(points, info.vertices, info.faces, origin);
  return dists.sqrt() * signs.to(dists.dtype());
}
}  // namespace geomlib
