#pragma once

#include <torch/torch.h>

#include "generalized_projection_info.h"

namespace geomlib {
torch::Tensor PointTriMeshSignedDistance(torch::Tensor points,
                                         torch::Tensor origin,
                                         const TriangularProjectionInfo& info);
}  // namespace geomlib
