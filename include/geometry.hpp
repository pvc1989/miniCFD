// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef PVC_CFD_GEOMETRY_HPP_
#define PVC_CFD_GEOMETRY_HPP_

#include <array>
#include <initializer_list>

namespace pvc {
namespace cfd {

template <class Real, int kDim>
class Geometry {
 public:
  class Point;
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
 private:
  std::array<Real, kDim> xyz_;
};

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
