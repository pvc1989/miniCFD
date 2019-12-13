//  Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_POLYNOMIAL_LINE_HPP_
#define MINI_POLYNOMIAL_LINE_HPP_

#include "mini/polynomial/legendre.hpp"

namespace mini {
namespace projector {
template <int kDegree> class Line;
}
namespace polynomial {

template <int kDegree>
class Line {
 public:
  using Column = algebra::Column<double, kDegree+1>;
  using Projector = projector::Line<kDegree>;
  Line(Projector* projector, Column coefficients)
      : projector_(projector), coefficients_(coefficients) {}
  double GetValueAtlocal(double x) const {
    auto values = Legendre<kDegree>::GetAllValues(x);
    return values.Dot(coefficients_);
  }
  double GetValueAtGlobal(double x) const {
    auto x_local = projector_->GlobalToLocal(x);
    return GetValueAtlocal(x_local);
  }

 private:
  Projector* projector_;
  Column coefficients_;
};

}  // namespace polynomial
}  // namespace mini

#endif  // MINI_POLYNOMIAL_LINE_HPP_
