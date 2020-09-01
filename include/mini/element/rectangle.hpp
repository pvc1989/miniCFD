// Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_ELEMENT_RECTANGLE_HPP_
#define MINI_ELEMENT_RECTANGLE_HPP_

#include <cstddef>
#include <initializer_list>

#include "mini/element/point.hpp"
#include "mini/element/surface.hpp"
#include "mini/geometry/rectangle.hpp"

namespace mini {
namespace element {

template <class Real, int kDim>
class Rectangle :
    virtual public Surface<Real, kDim>,
    public geometry::Rectangle<Real, kDim> {

 public:
  // Types:
  using IdType = typename Surface<Real, kDim>::IdType;
  using PointType = typename Surface<Real, kDim>::PointType;
  // Constructors:
  Rectangle(IdType i,
            const PointType& a, const PointType& b,
            const PointType& c, const PointType& d)
      : i_(i), geometry::Rectangle<Real, kDim>(a, b, c, d) {}
  // Accessors:
  IdType I() const override { return i_; }

 private:
  IdType i_;
};

}  // namespace element
}  // namespace mini

#endif  // MINI_ELEMENT_RECTANGLE_HPP_
