// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_GEOMETRY_LINE_HPP_
#define MINI_GEOMETRY_LINE_HPP_

#include <stdexcept>

#include "mini/geometry/point.hpp"
#include "mini/geometry/vector.hpp"

namespace mini {
namespace geometry {

template <class Real, int kDim>
class Line {
 public:
  // Types:
  using Point = Point<Real, kDim>;
  using Vector = Vector<Real, kDim>;

 public:
  // Constructors:
  Line(Point const& head, Point const& tail) : head_(&head), tail_(&tail) {}
  // Accessors:
  const Point& Head() const { return *head_ptr_; }
  const Point& Tail() const { return *tail_ptr_; }
  static int CountPoints() { return 2; }
  Point const& GetPoint(int i) const {
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
    Vector v = Head() - Tail();
    return std::sqrt(v.Dot(v));
  }
  Point Center() const {
    Point center = Head();
    center += Tail();
    center *= 0.5;
    return center;
  }

 private:
  Point* head_ptr_;
  Point* tail_ptr_;
};

}  // namespace geometry
}  // namespace mini

#endif  // MINI_GEOMETRY_LINE_HPP_
