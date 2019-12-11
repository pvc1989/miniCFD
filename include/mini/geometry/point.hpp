// Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_GEOMETRY_POINT_HPP_
#define MINI_GEOMETRY_POINT_HPP_

#include <array>
#include <initializer_list>
#include <cmath>
#include <utility>

#include "mini/algebra/column.hpp"

namespace mini {
namespace geometry {

template <class Real, int kDim>
class Vector;
template <class Real, int kDim>
class Point : public algebra::Column<Real, kDim> {
  using Base = algebra::Column<Real, kDim>;

 public:
  // Constructors:
  using Base::Base;
  explicit Point(Base const& that) : Base(that.begin(), that.end()) {}

  // Accessors:
  template <int I>
  Real X() const {
    return I < kDim ? this->at(I) : 0;
  }
  Real X() const { return X<0>(); }
  Real Y() const { return X<1>(); }
  Real Z() const { return X<2>(); }
  // Predicates:
  bool IsClockWise(Point const& b, Point const& c) const {
    static_assert(kDim == 2);
    return (b - *this).Cross(c - *this) < 0;
  }
  bool IsClockWise(Point const* b, Point const* c) const {
    return IsClockWise(*b, *c);
  }
};
// Binary operators:
template <class Real, int kDim>
Point<Real, kDim> operator+(
    Point<Real, kDim> const& lhs,
    Point<Real, kDim> const& rhs) {
  auto v = lhs;
  v += rhs;
  return v;
}
template <class Real, int kDim>
Vector<Real, kDim> operator-(
    Point<Real, kDim> const& lhs,
    Point<Real, kDim> const& rhs) {
  auto v = Vector<Real, kDim>(lhs);
  v -= rhs;
  return v;
}
template <class Scalar, int kSize>
Point<Scalar, kSize> operator*(
    Point<Scalar, kSize> const& lhs,
    Scalar const& rhs) {
  auto v = lhs;
  v *= rhs;
  return v;
}
template <class Scalar, int kSize>
Point<Scalar, kSize> operator/(
    Point<Scalar, kSize> const& lhs,
    Scalar const& rhs) {
  auto v = lhs;
  v /= rhs;
  return v;
}

}  // namespace geometry
}  // namespace mini
#endif  // MINI_GEOMETRY_POINT_HPP_
