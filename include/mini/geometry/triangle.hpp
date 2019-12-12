// Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_GEOMETRY_TRIANGLE_HPP_
#define MINI_GEOMETRY_TRIANGLE_HPP_

#include <stdexcept>

#include "mini/geometry/point.hpp"
#include "mini/geometry/surface.hpp"
#include "mini/geometry/vector.hpp"

namespace mini {
namespace geometry {

template <class Real, int kDim>
class Triangle : virtual public Surface<Real, kDim> {
 public:
  // Types:
  using Point = Point<Real, kDim>;
  // Constructors:
  Triangle(Point const& a, Point const& b, Point const& c)
      : a_(&a), b_(&b), c_(&c) {}
  // Accessors:
  int CountVertices() const override { return 3; }
  Point const& A() const { return *a_; }
  Point const& B() const { return *b_; }
  Point const& C() const { return *c_; }
  Point const& GetPoint(int i) const override {
    switch (i)  {
    case 0:
      return A();
    case 1:
      return B();
    case 2:
      return C();
    default:
      throw std::out_of_range("A `Triangle` has 3 `Point`s.");
    }
  }
  // Geometric methods:
  Real Measure() const override {
    auto v = (B() - A()).Cross(C() - A());
    return std::abs(v) * 0.5;
  }
  Point Center() const override {
    auto center = A();
    center += B();
    center += C();
    center /= 3.0;
    return center;
  }

 private:
  Point const* a_;
  Point const* b_;
  Point const* c_;
};

}  // namespace geometry
}  // namespace mini
#endif  // MINI_GEOMETRY_TRIANGLE_HPP_
