// Copyright 2019 PEI Weicheng and YANG Minghao

#ifndef MINI_ELEMENT_TRIANGLE_HPP_
#define MINI_ELEMENT_TRIANGLE_HPP_

#include <cstddef>
#include <initializer_list>

#include "mini/element/point.hpp"
#include "mini/element/surface.hpp"
#include "mini/geometry/triangle.hpp"

namespace mini {
namespace element {

template <class Real, int kDimensions>
class Triangle :
    virtual public Surface<Real, kDimensions>,
    public geometry::Triangle<Real, kDimensions> {
 public:
  // Types:
  using IdType = typename Surface<Real, kDimensions>::IdType;
  using PointType = typename Surface<Real, kDimensions>::PointType;
  // Constructors:
  Triangle(IdType i,
           const PointType& a, const PointType& b, const PointType& c)
      : i_(i), geometry::Triangle<Real, kDimensions>(a, b, c) {}
  // Accessors:
  IdType I() const override { return i_; }

 private:
  IdType i_;
};

}  // namespace element
}  // namespace mini

#endif  // MINI_ELEMENT_TRIANGLE_HPP_
