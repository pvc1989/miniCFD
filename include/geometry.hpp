// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef PVC_CFD_GEOMETRY_HPP_
#define PVC_CFD_GEOMETRY_HPP_

#include <array>
#include <initializer_list>
#include <utility>

namespace pvc {
namespace cfd {

template <class Real, int kDim>
class Geometry {
 public:
  class Point;
  class Vector;
  class Line;
};

template <class Real, int kDim>
class Geometry<Real, kDim>::Point {
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
    static_assert(0 <= I and I < kDim);
    return xyz_[I];
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
    if (this != &that) {
      for (auto i = 0; i != kDim; ++i) {
        point.xyz_[i] += that.xyz_[i];
      }
    }
    return point;
  }
  Vector operator-(const Point& that) const {
    auto point = Point(xyz_.begin(), xyz_.end());
    if (this != &that) {
      for (auto i = 0; i != kDim; ++i) {
        point.xyz_[i] -= that.xyz_[i];
      }
    }
    return point;
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
 private:
  std::array<Real, kDim> xyz_;
};

template <class Real, int kDim>
class Geometry<Real, kDim>::Vector : public Geometry<Real, kDim>::Point {
 public:
  // Constructors (forward to Point's constructors):
  template <class... T>
  Vector(T&&... t) : Point{std::forward<T>(t)...} {}
  // Operators:
  Real Dot(const Vector& that) {
    Real dot = 0.0;
    for (auto i = 0; i != kDim; ++i) {
      dot += xyz_[i] * that.xyz_[i];
    }
    return dot;
  }
  Vector Cross(const Vector& that) {
    assert(kDim >= 2);
    std::array<Real, 3> xyz;
    if (kDim == 2) {
      xyz[0] = 0.0;
      xyz[1] = 0.0;
      xyz[2] = xyz_[0] * that.xyz_[1] - xyz_[1] * that.xyz_[0];
    }
    else if (kDim == 3) {
      xyz[0] = xyz_[1] * that.xyz_[2] - xyz_[2] * that.xyz_[1];
      xyz[1] = xyz_[2] * that.xyz_[0] - xyz_[0] * that.xyz_[2];
      xyz[2] = xyz_[0] * that.xyz_[1] - xyz_[1] * that.xyz_[0];
    }
    auto cross = Vector(xyz.begin(), xyz.end());
    return cross;
  }
};

template <class Real>
typename Geometry<Real, 3>::Vector CrossProduct(
    typename Geometry<Real, 3>::Vector const& lhs,
    typename Geometry<Real, 3>::Vector const& rhs) {
  return rhs;
}
template <class Real>
Real CrossProduct(
    typename Geometry<Real, 2>::Vector const& lhs,
    typename Geometry<Real, 2>::Vector const& rhs) {
  return 0;
}

template <class Real, int kDim>
class Geometry<Real, kDim>::Line {
 public:
  Line(Point* head, Point* tail) : head_(head), tail_(tail) {}
  Point* Head() const { return head_; }
  Point* Tail() const { return tail_; }
 private:
  Point* head_{nullptr};
  Point* tail_{nullptr};
};

}  // namespace cfd
}  // namespace pvc
#endif  // PVC_CFD_GEOMETRY_HPP_
