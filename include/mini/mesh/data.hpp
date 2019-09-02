// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_MESH_DATA_HPP_
#define MINI_MESH_DATA_HPP_

#include <array>
#include <string>

namespace mini {
namespace mesh {

struct Empty {
  static constexpr int CountScalars() { return 0; }
  static constexpr int CountVectors() { return 0; }
};

template <class Real, int kDimensions, int kScalars, int kVectors>
struct Data {
 public:
  using Scalar = Real;
  using Vector = std::array<Real, kDimensions>;
  static constexpr int CountScalars() { return kScalars; }
  static constexpr int CountVectors() { return kVectors; }
  std::array<Scalar, kScalars> scalars;
  std::array<Vector, kVectors> vectors;
};

}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_DATA_HPP_
