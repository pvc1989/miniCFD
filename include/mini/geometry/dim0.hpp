// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_GEOMETRY_DIM0_HPP_
#define MINI_GEOMETRY_DIM0_HPP_

#include <array>
#include <initializer_list>
#include <cmath>
#include <utility>

namespace std {
template <class Vector>
auto abs(Vector const& v) {
  return std::sqrt(v.Dot(v));
}
}  // namespace std

namespace mini {
namespace geometry {

template <class Real, int kDim>
class Vector;
template <class Real, int kDim>
class Point {
 public:
  // Constructors:
  template <class Iterator>
  Point(Iterator first, Iterator last) {
    assert(last - first == kDim);
    auto curr = xyz_.begin();
    while (first != last) { *curr++ = *first++; }
    assert(curr = xyz_.end());
  }
  Point(std::initializer_list<Real> xyz) : Point(xyz.begin(), xyz.end()) {}
  // Accessors:
  template <int I>
  Real X() const {
    return I < kDim ? xyz_[I] : 0;
  }
  Real X() const { return X<0>(); }
  Real Y() const { return X<1>(); }
  Real Z() const { return X<2>(); }
  // Operators:
  Point operator=(const Point& that) const {
    return Point(that.xyz_.begin(), that.xyz_.end());
  }
  Point operator+(const Point& that) const {
    auto point = Point(xyz_.begin(), xyz_.end());
    for (auto i = 0; i != kDim; ++i) {
      point.xyz_[i] += that.xyz_[i];
    }
    return point;
  }
  Vector<Real, kDim> operator-(const Point& that) const {
    auto v = Vector<Real, kDim>(xyz_.begin(), xyz_.end());
    for (auto i = 0; i != kDim; ++i) {
      v.xyz_[i] -= that.xyz_[i];
    }
    return v;
  }
  Point operator*(const Real& scalar) const {
    auto point = Point(xyz_.begin(), xyz_.end());
    for (auto i = 0; i != kDim; ++i) {
      point.xyz_[i] *= scalar;
    }
    return point;
  }
  Point operator/(const Real& scalar) const {
    assert(scalar != 0);
    auto point = Point(xyz_.begin(), xyz_.end());
    for (auto i = 0; i != kDim; ++i) {
      point.xyz_[i] /= scalar;
    }
    return point;
  }

 protected:
  std::array<Real, kDim> xyz_;
};

// Non-member operators:
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

template <class Real, int kDim>
class Vector : public Point<Real, kDim> {
 public:
  // Constructors (forward to Point's constructors):
  template <class... T>
  explicit Vector(T&&... t) : Point<Real, kDim>{std::forward<T>(t)...} {}
  // Operators:
  Real Dot(const Vector& that) const {
    Real dot = 0.0;
    for (auto i = 0; i != kDim; ++i) {
      dot += this->xyz_[i] * that.xyz_[i];
    }
    return dot;
  }
  auto Cross(const Vector& that) const {
    static_assert(kDim == 2 || kDim == 3);
    return CrossProduct<Real>(*this, that);
  }
};

}  // namespace geometry
}  // namespace mini
#endif  // MINI_GEOMETRY_DIM0_HPP_
