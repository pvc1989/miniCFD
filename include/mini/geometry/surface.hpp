// Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_GEOMETRY_SURFACE_HPP_
#define MINI_GEOMETRY_SURFACE_HPP_

#include "mini/geometry/point.hpp"

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

}  // namespace geometry
}  // namespace mini
#endif  // MINI_GEOMETRY_SURFACE_HPP_
