// Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_ELEMENT_POINT_HPP_
#define MINI_ELEMENT_POINT_HPP_

#include <cstddef>
#include <initializer_list>
#include <utility>

#include "mini/geometry/point.hpp"

namespace mini {
namespace element {

template <class Real, int kDim>
class Point : public geometry::Point<Real, kDim> {
 public:
  // Types:
  using Id = std::size_t;
  // Constructors:
  template<class... XYZ>
  Point(Id i, XYZ&&... xyz)
      : i_(i), geometry::Point<Real, kDim>{std::forward<XYZ>(xyz)...} {}
  Point(Id i, std::initializer_list<Real> xyz)
      : i_(i), geometry::Point<Real, kDim>(xyz) {}
  Point(std::initializer_list<Real> xyz) : Point{DefaultId(), xyz} {}
  // Accessors:
  Id I() const { return i_; }
  static Id DefaultId() { return -1; }
 private:
  Id i_;
};

}  // namespace element
}  // namespace mini

#endif  // MINI_ELEMENT_POINT_HPP_
