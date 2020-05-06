// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_ELEMENT_LINE_HPP_
#define MINI_ELEMENT_LINE_HPP_

#include <cstddef>
#include <initializer_list>

#include "mini/geometry/line.hpp"
#include "mini/element/point.hpp"

namespace mini {
namespace element {

template <class Real, int kDim>
class Line : public geometry::Line<Real, kDim> {
  using Base = geometry::Line<Real, kDim>;

 public:
  // Types:
  using IndexType = std::size_t;
  using PointType = Point<Real, kDim>;
  // Constructors:
  Line(IndexType i, const PointType& head, const PointType& tail)
      : i_(i), Base(head, tail) {}
  // Accessors:
  Line::IndexType I() const { return i_; }
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
  IndexType i_;
};

}  // namespace element
}  // namespace mini

#endif  // MINI_ELEMENT_LINE_HPP_
