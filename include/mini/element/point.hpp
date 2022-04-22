// Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_ELEMENT_POINT_HPP_
#define MINI_ELEMENT_POINT_HPP_

#include <cstddef>
#include <initializer_list>
#include <utility>

#include "mini/geometry/point.hpp"

namespace mini {
namespace element {

template <class Real, int kDimensions>
class Point : public geometry::Point<Real, kDimensions> {
 public:
  // Types:
  using IdType = std::size_t;
  // Constructors:
  template<class... XYZ>
  explicit Point(IdType i, XYZ&&... xyz)
      : i_(i), geometry::Point<Real, kDimensions>{std::forward<XYZ>(xyz)...} {}
  Point(IdType i, std::initializer_list<Real> xyz)
      : i_(i), geometry::Point<Real, kDimensions>(xyz) {}
  // Accessors:
  IdType I() const { return i_; }

 private:
  IdType i_;
};

}  // namespace element
}  // namespace mini

#endif  // MINI_ELEMENT_POINT_HPP_
