//  Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_PROJECTOR_LINE_HPP_
#define MINI_PROJECTOR_LINE_HPP_

#include "mini/algebra/column.hpp"
#include "mini/geometry/point.hpp"
#include "mini/integrator/gauss.hpp"
#include "mini/polynomial/legendre.hpp"
#include "mini/polynomial/line.hpp"

namespace mini {
namespace projector {

template <int kDegree>
class Line {
 public:
  using Point = geometry::Point<double, 1>;
  Line(Point const& head, Point const& tail)
      : jacobian_((tail.X() - head.X()) / 2),
        x_center_((tail.X() + head.X()) / 2) {}
  using Column = algebra::Column<double, kDegree+1>;
  template <class Function>
  Column GetCoefficients(Function&& function) const {
    auto integrand = [&](double x) {
      auto value = polynomial::Legendre<kDegree>::GetAllValues(x);
      value *= function(LocalToGlobal(x));
      return value;
    };
    auto result = integrator::Gauss<kDegree+1>::Integrate(integrand);
    for (int i = 0; i <= kDegree; i++) {
      result[i] *= polynomial::normalizer[i];
    }
    return result;
  }
  using Polynomial = polynomial::Line<kDegree>;
  template <class Function>
  Polynomial GetApproximate(Function&& function) const {
    auto coefficients = GetCoefficients(function);
    return Polynomial(this, coefficients);
  }
 
 private:
  double jacobian_;
  double x_center_;
  auto LocalToGlobal(double x) const {
    return x * jacobian_ + x_center_;
  }
  auto GlobalToLocal(double x) const {
    return (x - x_center_) / jacobian_;
  }
};

}  // namespace projector
}  // namespace mini

#endif  // MINI_PROJECTOR_LINE_HPP_
