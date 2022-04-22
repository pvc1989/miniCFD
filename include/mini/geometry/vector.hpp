// Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_GEOMETRY_VECTOR_HPP_
#define MINI_GEOMETRY_VECTOR_HPP_

#include "mini/geometry/point.hpp"

namespace mini {
namespace geometry {

template <class Real, int kDimensions>
class Vector;

template <class Real, int kDimensions>
auto CrossProduct(Vector<Real, kDimensions> const& lhs, Vector<Real, kDimensions> const& rhs);

template <class Real, int kDimensions>
class Vector : public Point<Real, kDimensions> {
  using Base = Point<Real, kDimensions>;

 public:
  // Constructors:
  using Base::Base;
  explicit Vector(Base const& that) : Base(that) {}
  // Operators:
  auto Cross(Vector const& that) const {
    static_assert(kDimensions == 2 || kDimensions == 3);
    return CrossProduct<Real>(*this, that);
  }
};
template <class Real, int kDimensions>
Vector<Real, kDimensions> operator+(
    Vector<Real, kDimensions> const& lhs,
    Vector<Real, kDimensions> const& rhs) {
  auto v = lhs;
  v += rhs;
  return v;
}
// Implement CrossProduct()
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

// Point Methods
template <class Real, int kDimensions>
Vector<Real, kDimensions> operator-(
    Point<Real, kDimensions> const& lhs,
    Point<Real, kDimensions> const& rhs) {
  auto v = Vector<Real, kDimensions>(lhs);
  v -= rhs;
  return v;
}
// Predicates:
template <class Real>
bool IsClockWise(Point<Real, 2> const& a,
                 Point<Real, 2> const& b,
                 Point<Real, 2> const& c) {
  // The difference of two `Point`s is a `Vector`, which has a `Cross` method.
  return (b - a).Cross(c - a) < 0;
}

}  // namespace geometry
}  // namespace mini
#endif  // MINI_GEOMETRY_VECTOR_HPP_
