// Copyright 2019 PEI Weicheng and YANG Minghao

#ifndef MINI_GEOMETRY_LINE_HPP_
#define MINI_GEOMETRY_LINE_HPP_

#include <stdexcept>

#include "mini/geometry/point.hpp"
#include "mini/geometry/vector.hpp"

namespace mini {
namespace geometry {

template <class Real, int kDimensions>
class Line {
 public:
  // Types:
  using PointType = Point<Real, kDimensions>;

 public:
  // Constructors:
  Line(const PointType& head, const PointType& tail)
      : head_(head), tail_(tail) {}
  // Accessors:
  const PointType& Head() const { return head_; }
  const PointType& Tail() const { return tail_; }
  static int CountPoints() { return 2; }
  const PointType& GetPoint(int i) const {
    switch (i)  {
    case 0:
      return Head();
    case 1:
      return Tail();
    default:
      throw std::out_of_range("A `Line` has two `Point`s.");
    }
  }
  // Geometric methods:
  Real Measure() const {
    auto v = Head() - Tail();
    return std::sqrt(v.Dot(v));
  }
  PointType Center() const {
    auto center = Head();
    center += Tail();
    center *= 0.5;
    return center;
  }

 private:
  const PointType& head_;
  const PointType& tail_;
};

}  // namespace geometry
}  // namespace mini

#endif  // MINI_GEOMETRY_LINE_HPP_
