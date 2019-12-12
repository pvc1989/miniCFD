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
  using Id = std::size_t;
  using Point = Point<Real, kDim>;
  // Constructors:
  Line(Id i, Point const& head, Point const& tail)
      : i_(i), geometry::Line<Real, kDim>(head, tail) {}
  Line(Point const& head, Point const& tail)
      : Line(DefaultId(), head, tail) {}
  // Accessors:
  Line::Id I() const { return i_; }
  static Id DefaultId() { return -1; }
  // Up-casts:
  auto Head() const {
    return reinterpret_cast<Point const&>(Base::Head());
  }
  auto Tail() const {
    return reinterpret_cast<Point const&>(Base::Tail());
  }
  auto GetPoint(int i) const {
    return reinterpret_cast<Point const&>(Base::GetPoint(i));
  }
  // Mesh methods:
  template <class Integrand>
  auto Integrate(Integrand&& integrand) const {
    return integrand(this->Center()) * this->Measure();
  }

 private:
  Id i_;
};

}  // namespace element
}  // namespace mini

#endif  // MINI_ELEMENT_LINE_HPP_
