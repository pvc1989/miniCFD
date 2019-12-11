// Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_GEOMETRY_VECTOR_HPP_
#define MINI_GEOMETRY_VECTOR_HPP_

#include <array>
#include <initializer_list>
#include <cmath>
#include <utility>

#include "mini/geometry/point.hpp"

namespace mini {
namespace geometry {

template <class Real, int kDim>
class Vector;

template <class Real, int kDim>
auto CrossProduct(Vector<Real, kDim> const& lhs, Vector<Real, kDim> const& rhs);

template <class Real, int kDim>
class Vector : public Point<Real, kDim> {
  using Base = Point<Real, kDim>;

 public:
  // Constructors:
  using Base::Base;
  explicit Vector(Base const& that) : Base(that.begin(), that.end()) {}
  // Operators:
  auto Cross(Vector const& that) const {
    static_assert(kDim == 2 || kDim == 3);
    return CrossProduct<Real>(*this, that);
  }
};
template <class Real, int kDim>
Vector<Real, kDim> operator+(
    Vector<Real, kDim> const& lhs,
    Vector<Real, kDim> const& rhs) {
  auto v = lhs;
  v += rhs;
  return v;
}
template <class Real>
auto CrossProduct(Vector<Real, 3> const& lhs, Vector<Real, 3> const& rhs) {
  auto x = lhs.Y() * rhs.Z() - lhs.Z() * rhs.Y();
  auto y = lhs.Z() * rhs.X() - lhs.X() * rhs.Z();
  auto z = lhs.X() * rhs.Y() - lhs.Y() * rhs.X();
  return Vector<Real, 3>{x, y, z};
}
template <class Real>
auto CrossProduct(Vector<Real, 2> const& lhs, Vector<Real, 2> const& rhs) {
  return lhs.X() * rhs.Y() - lhs.Y() * rhs.X();
}

}  // namespace geometry
}  // namespace mini
#endif  // MINI_GEOMETRY_VECTOR_HPP_
