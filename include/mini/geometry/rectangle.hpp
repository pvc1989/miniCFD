// Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_GEOMETRY_RECTANGLE_HPP_
#define MINI_GEOMETRY_RECTANGLE_HPP_

#include <stdexcept>

#include "mini/geometry/point.hpp"
#include "mini/geometry/surface.hpp"
#include "mini/geometry/vector.hpp"

namespace mini {
namespace geometry {

template <class Real, int kDimensions>
class Rectangle : virtual public Surface<Real, kDimensions> {
 protected:
  using Base = Surface<Real, kDimensions>;

 public:
  // Types:
  using PointType = typename Base::PointType;
  // Constructors:
  Rectangle(const PointType& a, const PointType& b,
            const PointType& c, const PointType& d)
      : a_(a), b_(b), c_(c) , d_(d) {}
  // Accessors:
  int CountVertices() const override { return 4; }
  const PointType& A() const { return a_; }
  const PointType& B() const { return b_; }
  const PointType& C() const { return c_; }
  const PointType& D() const { return d_; }
  const PointType& GetPoint(int i) const override {
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
  PointType Center() const override {
    auto center = A();
    center += C();
    center *= 0.5;
    return center;
  }

 private:
  const PointType& a_;
  const PointType& b_;
  const PointType& c_;
  const PointType& d_;
};

}  // namespace geometry
}  // namespace mini
#endif  // MINI_GEOMETRY_RECTANGLE_HPP_
