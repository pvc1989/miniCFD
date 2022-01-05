//  Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_RIEMANN_LINEAR_SINGLE_HPP_
#define MINI_RIEMANN_LINEAR_SINGLE_HPP_

#include <cmath>
#include <array>

#include "mini/algebra/eigen.hpp"

namespace mini {
namespace riemann {
namespace linear {

template <int D>
class Single {
 public:
  static constexpr int kFunc = 1;
  static constexpr int kDim = D;
  // Types:
  using Scalar = double;
  using Vector = algebra::Vector<double, kDim>;
  using Jacobi = double;
  using Coefficient = algebra::Vector<Jacobi, kDim>;
  using Conservative = double;
  using Flux = double;
  using Speed = double;
  using MatKx1 = algebra::Matrix<Scalar, kFunc, 1>;
  // Constructor:
  Single() : a_const_(1) {}
  explicit Single(Jacobi const& a_const) : a_const_(a_const) {}
  // Get F on T Axia:
  Flux GetFluxOnTimeAxis(const Conservative& left, const Conservative& right) const {
    if (0 < a_const_) {
      return left * a_const_;
    } else {
      return right* a_const_;
    }
  }
  Flux GetFluxOnTimeAxis(const MatKx1& left, const MatKx1& right) const {
    return GetFluxOnTimeAxis(left[0], right[0]);
  }
  // Get F of U
  Flux GetFlux(const Conservative& state) const {
    return state * a_const_;
  }
  Flux GetFlux(const MatKx1& state) const {
    return GetFlux(state[0]);
  }

 private:
  Jacobi a_const_;
};

}  // namespace linear
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_LINEAR_SINGLE_HPP_
