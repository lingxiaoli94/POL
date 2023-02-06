#include "triangular_mesh_sampler.h"

#include <ATen/Functions.h>

using namespace torch::indexing;

namespace geomlib {
TriangularMeshSampler::TriangularMeshSampler(torch::Tensor vertices,
                                             torch::Tensor faces)
    : vertices_{vertices}, faces_{faces} {
  auto v0 = vertices.index({faces.index({Slice(), 0}), Slice()});
  auto v1 = vertices.index({faces.index({Slice(), 1}), Slice()});
  auto v2 = vertices.index({faces.index({Slice(), 2}), Slice()});
  auto area_vec = (v1 - v0).cross(v2 - v0);
  areas_ = (area_vec.square().sum(-1) + 1e-8f).sqrt() / 2;
}

torch::Tensor TriangularMeshSampler::Sample(int batch_size) {
  return SampleWithWeights(batch_size)[0];
}

std::vector<torch::Tensor> TriangularMeshSampler::SampleWithWeights(
    int batch_size) {
  // Sample with replacement.
  auto chosen_faces = torch::multinomial(areas_ + 1e-8f, batch_size, true);
  auto v0 = vertices_.index({faces_.index({chosen_faces, 0}), Slice()});
  auto v1 = vertices_.index({faces_.index({chosen_faces, 1}), Slice()});
  auto v2 = vertices_.index({faces_.index({chosen_faces, 2}), Slice()});
  auto vs = torch::stack({v0, v1, v2}, 2);  // Bx3x3

  auto r1 = torch::rand({batch_size}, v0.options());
  auto r2 = torch::rand({batch_size}, v0.options());

  auto weights = torch::stack(
      {1 - torch::sqrt(r1 + 1e-10f), torch::sqrt(r1 + 1e-10f) * (1 - r2),
       r2 * torch::sqrt(r1 + 1e-10f)},
      1);  // Bx3
  auto samples = (weights.unsqueeze(1) * vs).sum(-1);
  return {samples, chosen_faces, weights};
}

torch::Tensor TriangularMeshSampler::GetVertices() { return vertices_; }

torch::Tensor TriangularMeshSampler::GetFaces() { return faces_; }
}  // namespace geomlib
