//  Copyright 2019 PEI Weicheng and YANG Minghao

#ifndef MINI_RIEMANN_LINEAR_SINGLE_HPP_
#define MINI_RIEMANN_LINEAR_SINGLE_HPP_

#include <cmath>
#include <array>

#include "mini/algebra/eigen.hpp"

namespace mini {
namespace riemann {
namespace linear {

template <typename S, int D>
class Single {
 public:
  static constexpr int kComponents = 1;
  static constexpr int kDimensions = D;
  // Types:
  using Scalar = S;
  using Vector = algebra::Vector<Scalar, kDimensions>;
  using Jacobi = Scalar;
  using Coefficient = algebra::Vector<Jacobi, kDimensions>;
  using Conservative = Scalar;
  using Flux = Scalar;
  using Speed = Scalar;
  using MatKx1 = algebra::Matrix<Scalar, kComponents, 1>;
  // Constructor:
  Single() : a_const_(1) {}
  explicit Single(Jacobi const& a_const) : a_const_(a_const) {}
  // Get F on T Axia:
  Flux GetFluxOnTimeAxis(const Conservative& left, const Conservative& right)
      const {
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
