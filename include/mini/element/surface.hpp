// Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_ELEMENT_SURFACE_HPP_
#define MINI_ELEMENT_SURFACE_HPP_

#include <cstddef>
#include <initializer_list>

#include "mini/element/point.hpp"
#include "mini/geometry/surface.hpp"

namespace mini {
namespace element {

template <class Real, int kDim>
class Surface : virtual public geometry::Surface<Real, kDim> {
 public:
  // Types:
  using IdType = std::size_t;
  using PointType = Point<Real, kDim>;
  // Accessors:
  virtual IdType I() const = 0;
  // Mesh methods:
  template <class Integrand>
  auto Integrate(Integrand&& integrand) const {
    return integrand(this->Center()) * this->Measure();
  }
};

}  // namespace element
}  // namespace mini

#endif  // MINI_ELEMENT_SURFACE_HPP_
