// Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_GEOMETRY_TRIANGLE_HPP_
#define MINI_GEOMETRY_TRIANGLE_HPP_

#include "mini/geometry/point.hpp"
#include "mini/geometry/line.hpp"
#include "mini/geometry/vector.hpp"

namespace mini {
namespace geometry {

template <class Real, int kDim>
class Surface {
 public:
  // Types:
  using Point = Point<Real, kDim>;
  // Accessors:
  virtual int CountVertices() const = 0;
  virtual Point* GetPoint(int i) const = 0;
  // Geometric methods:
  virtual Real Measure() const = 0;
  virtual Point Center() const = 0;
};

template <class Real, int kDim>
class Triangle : virtual public Surface<Real, kDim> {
 public:
  // Types:
  using Point = Point<Real, kDim>;
  // Constructors:
  Triangle(Point* a, Point* b, Point* c) : a_(a), b_(b), c_(c) {}
  // Accessors:
  int CountVertices() const override { return 3; }
  Point* GetPoint(int i) const override {
    switch (i)  {
    case 0:
      return a_;
    case 1:
      return b_;
    case 2:
      return c_;
    default:
      return nullptr;
    }
  }
  // Geometric methods:
  Real Measure() const override {
    auto v = (*b_ - *a_).Cross(*c_ - *a_);
    return std::abs(v) * 0.5;
  }
  Point Center() const override {
    auto center = *a_;
    center += *b_;
    center += *c_;
    center /= 3.0;
    return center;
  }

 private:
  Point* a_;
  Point* b_;
  Point* c_;
};

template <class Real, int kDim>
class Rectangle : virtual public Surface<Real, kDim> {
 public:
  // Types:
  using Point = Point<Real, kDim>;
  // Constructors:
  Rectangle(Point* a, Point* b, Point* c, Point* d)
      : a_(a), b_(b), c_(c) , d_(d) {}
  // Accessors:
  int CountVertices() const override { return 4; }
  Point* GetPoint(int i) const override {
    switch (i)  {
    case 0:
      return a_;
    case 1:
      return b_;
    case 2:
      return c_;
    case 3:
      return d_;
    default:
      return nullptr;
    }
  }
  // Geometric methods:
  Real Measure() const override {
    auto v = (*b_ - *a_).Cross(*c_ - *a_);
    return std::abs(v);
  }
  Point Center() const override {
    auto center = *a_;
    center += *c_;
    center *= 0.5;
    return center;
  }

 private:
  Point* a_;
  Point* b_;
  Point* c_;
  Point* d_;
};

}  // namespace geometry
}  // namespace mini
#endif  // MINI_GEOMETRY_TRIANGLE_HPP_
