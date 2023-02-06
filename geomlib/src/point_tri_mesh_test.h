#pragma once

#include <torch/torch.h>

namespace geomlib {
// Returned tensor contains 1 or -1 (type same as faces).
torch::Tensor PointTriMeshTest(torch::Tensor points, torch::Tensor vertices,
                               torch::Tensor faces, torch::Tensor origin);
}  // namespace geomlib
