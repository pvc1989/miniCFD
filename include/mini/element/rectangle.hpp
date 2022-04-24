// Copyright 2019 PEI Weicheng and YANG Minghao
#ifndef MINI_ELEMENT_RECTANGLE_HPP_
#define MINI_ELEMENT_RECTANGLE_HPP_

#include <cstddef>
#include <initializer_list>

#include "mini/element/point.hpp"
#include "mini/element/surface.hpp"
#include "mini/geometry/rectangle.hpp"

namespace mini {
namespace element {

template <class Real, int kDimensions>
class Rectangle :
    virtual public Surface<Real, kDimensions>,
    public geometry::Rectangle<Real, kDimensions> {
 public:
  // Types:
  using IdType = typename Surface<Real, kDimensions>::IdType;
  using PointType = typename Surface<Real, kDimensions>::PointType;
  // Constructors:
  Rectangle(IdType i,
            const PointType& a, const PointType& b,
            const PointType& c, const PointType& d)
      : i_(i), geometry::Rectangle<Real, kDimensions>(a, b, c, d) {}
  // Accessors:
  IdType I() const override { return i_; }

 private:
  IdType i_;
};

}  // namespace element
}  // namespace mini

#endif  // MINI_ELEMENT_RECTANGLE_HPP_
