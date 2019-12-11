//  Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_PROJECTOR_LINE_HPP_
#define MINI_PROJECTOR_LINE_HPP_

#include "mini/algebra/column.hpp"
#include "mini/integrator/gauss.hpp"
#include "mini/polynomial/legendre.hpp"

namespace mini {
namespace projector {

template <int kDegree>
class Line {
 public:
  Line(Point const& head, Point const& tail) {
  }
  using Vector = algebra::Column<double, kDegree+1>;
  template <class Function>
  Vector GetCoefficients(Function&& function) const {
  }
};

}  // namespace projector
}  // namespace mini

#endif  // MINI_PROJECTOR_LINE_HPP_
