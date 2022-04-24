// Copyright 2019 PEI Weicheng and YANG Minghao
#ifndef MINI_GEOMETRY_SURFACE_HPP_
#define MINI_GEOMETRY_SURFACE_HPP_

#include "mini/geometry/point.hpp"

namespace mini {
namespace geometry {

template <class Real, int kDimensions>
class Surface {
 public:
  // Types:
  using PointType = Point<Real, kDimensions>;
  // Accessors:
  virtual int CountVertices() const = 0;
  virtual const PointType& GetPoint(int i) const = 0;
  // Geometric methods:
  virtual Real Measure() const = 0;
  virtual PointType Center() const = 0;
};

}  // namespace geometry
}  // namespace mini
#endif  // MINI_GEOMETRY_SURFACE_HPP_
