// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_ELEMENT_DATA_HPP_
#define MINI_ELEMENT_DATA_HPP_

#include <array>
#include <string>

namespace mini {
namespace element {

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

using Empty = Data<int, 0, 0, 0>;

}  // namespace element
}  // namespace mini

#endif  // MINI_ELEMENT_DATA_HPP_
