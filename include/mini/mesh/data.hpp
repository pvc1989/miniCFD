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
  static std::array<std::string, 0> scalar_names;
  static std::array<std::string, 0> vector_names;
};
std::array<std::string, 0> Empty::scalar_names;
std::array<std::string, 0> Empty::vector_names;

template <class Real, int kDimensions, int kScalars, int kVectors>
struct Data {
 public:
  using Scalar = Real;
  using Vector = std::array<Real, kDimensions>;
  static constexpr int CountScalars() { return kScalars; }
  static constexpr int CountVectors() { return kVectors; }
  std::array<Scalar, kScalars> scalars;
  std::array<Vector, kVectors> vectors;
  static std::array<std::string, kScalars> scalar_names;
  static std::array<std::string, kVectors> vector_names;
};
template <class Real, int kDimensions, int kScalars, int kVectors>
std::array<std::string, kScalars> Data<Real, kDimensions, kScalars, kVectors>::scalar_names;
template <class Real, int kDimensions, int kScalars, int kVectors>
std::array<std::string, kVectors> Data<Real, kDimensions, kScalars, kVectors>::vector_names;

}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_DATA_HPP_
