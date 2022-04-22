// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_ELEMENT_LINE_HPP_
#define MINI_ELEMENT_LINE_HPP_

#include <cstddef>
#include <initializer_list>

#include "mini/geometry/line.hpp"
#include "mini/element/point.hpp"

namespace mini {
namespace element {

template <class Real, int kDimensions>
class Line : public geometry::Line<Real, kDimensions> {
  using Base = geometry::Line<Real, kDimensions>;

 public:
  // Types:
  using IdType = std::size_t;
  using PointType = Point<Real, kDimensions>;
  // Constructors:
  Line(IdType i, const PointType& head, const PointType& tail)
      : i_(i), Base(head, tail) {}
  // Accessors:
  Line::IdType I() const { return i_; }
  // Up-casts:
  const PointType& Head() const {
    return static_cast<const PointType&>(Base::Head());
  }
  const PointType& Tail() const {
    return static_cast<const PointType&>(Base::Tail());
  }
  // Mesh methods:
  template <class Integrand>
  auto Integrate(Integrand&& integrand) const {
    return integrand(this->Center()) * this->Measure();
  }

 private:
  IdType i_;
};

}  // namespace element
}  // namespace mini

#endif  // MINI_ELEMENT_LINE_HPP_
