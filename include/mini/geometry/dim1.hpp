// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_GEOMETRY_DIM1_HPP_
#define MINI_GEOMETRY_DIM1_HPP_

#include "mini/geometry/dim0.hpp"

namespace mini {
namespace geometry {

template <class Real, int kDim>
class Line {
 public:
  // Types:
  using Point = Point<Real, kDim>;
  // Constructors:
  Line(Point* head, Point* tail) : head_(head), tail_(tail) {}
  // Accessors:
  Point* Head() const { return head_; }
  Point* Tail() const { return tail_; }
  // Geometric methods:
  Real Measure() const {
    auto v = *head_ - *tail_;
    return std::sqrt(v.Dot(v));
  }
  Point Center() const {
    auto center = *head_;
    center += *tail_;
    center *= 0.5;
    return center;
  }
 private:
  Point* head_{nullptr};
  Point* tail_{nullptr};
};

}  // namespace geometry
}  // namespace mini
#endif  // MINI_GEOMETRY_DIM1_HPP_
