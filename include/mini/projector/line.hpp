//  Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_PROJECTOR_LINE_HPP_
#define MINI_PROJECTOR_LINE_HPP_

#include "mini/algebra/column.hpp"
#include "mini/geometry/dim0.hpp"
#include "mini/integrator/gauss.hpp"
#include "mini/polynomial/legendre.hpp"

namespace mini {
namespace projector {

template <int kDegree>
class Line {
 public:
  using Point = geometry::Point<double, 1>;
  Line(Point const& head, Point const& tail) {
    head_ = &head;
    tail_ = &tail;
    jacob_ = (tail.X() - head.X()) / 2;
    x_mid_ = (tail.X() + head.X()) / 2;
  }
  using Vector = algebra::Column<double, kDegree+1>;
  template <class Function>
  Vector GetCoefficients(Function&& function) const {
    auto integrand = [&](double x) {
      auto value = polynomial::Legendre<kDegree>::GetAllValues(x);
      value *= function(LocalToGlobal(x));
      return value;
    };
    auto result = integrator::Gauss<kDegree+1>::Integrate(integrand);
    for (int i = 0; i <= kDegree; i++) {
      result[i] *= polynomial::norms[i];
    }
    return result;
  }
 
 private:
  const Point* head_;
  const Point* tail_;
  double jacob_;
  double x_mid_;
  auto LocalToGlobal(double x) const {
    return x * jacob_ + x_mid_;
  }
  auto GlobalToLocal(double x) const {
    return (x - x_mid_) / jacob_;
  }
};

}  // namespace projector
}  // namespace mini

#endif  // MINI_PROJECTOR_LINE_HPP_
