// Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_GEOMETRY_RECTANGLE_HPP_
#define MINI_GEOMETRY_RECTANGLE_HPP_

#include <stdexcept>

#include "mini/geometry/point.hpp"
#include "mini/geometry/surface.hpp"
#include "mini/geometry/vector.hpp"

namespace mini {
namespace geometry {

template <class Real, int kDim>
class Rectangle : virtual public Surface<Real, kDim> {
 public:
  // Types:
  using Point = Point<Real, kDim>;
  // Constructors:
  Rectangle(Point const& a, Point const& b, Point const& c, Point const& d)
      : a_(&a), b_(&b), c_(&c) , d_(&d) {}
  // Accessors:
  int CountVertices() const override { return 4; }
  Point const& A() const { return *a_; }
  Point const& B() const { return *b_; }
  Point const& C() const { return *c_; }
  Point const& D() const { return *d_; }
  Point const& GetPoint(int i) const override {
    switch (i)  {
    case 0:
      return A();
    case 1:
      return B();
    case 2:
      return C();
    case 3:
      return D();
    default:
      throw std::out_of_range("A `Triangle` has 3 `Point`s.");
    }
  }
  // Geometric methods:
  Real Measure() const override {
    auto v = (B() - A()).Cross(C() - A());
    return std::abs(v);
  }
  Point Center() const override {
    auto center = A();
    center += C();
    center *= 0.5;
    return center;
  }

 private:
  Point const* a_;
  Point const* b_;
  Point const* c_;
  Point const* d_;
};

}  // namespace geometry
}  // namespace mini
#endif  // MINI_GEOMETRY_RECTANGLE_HPP_
