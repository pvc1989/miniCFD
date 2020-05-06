// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_ELEMENT_TRIANGLE_HPP_
#define MINI_ELEMENT_TRIANGLE_HPP_

#include <cstddef>
#include <initializer_list>

#include "mini/element/point.hpp"
#include "mini/element/surface.hpp"
#include "mini/geometry/triangle.hpp"

namespace mini {
namespace element {

template <class Real, int kDim>
class Triangle :
    virtual public Surface<Real, kDim>,
    public geometry::Triangle<Real, kDim> {

 public:
  // Types:
  using IndexType = typename Surface<Real, kDim>::IndexType;
  using PointType = typename Surface<Real, kDim>::PointType;
  // Constructors:
  Triangle(IndexType i,
           const PointType& a, const PointType& b, const PointType& c)
      : i_(i), geometry::Triangle<Real, kDim>(a, b, c) {}
  // Accessors:
  IndexType I() const override { return i_; }

 private:
  IndexType i_;
};

}  // namespace element
}  // namespace mini

#endif  // MINI_ELEMENT_TRIANGLE_HPP_
