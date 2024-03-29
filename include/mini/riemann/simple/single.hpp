//  Copyright 2019 PEI Weicheng and YANG Minghao

#ifndef MINI_RIEMANN_SIMPLE_SINGLE_HPP_
#define MINI_RIEMANN_SIMPLE_SINGLE_HPP_

#include <cmath>
#include <array>

#include "mini/algebra/eigen.hpp"

namespace mini {
namespace riemann {
namespace simple {

template <typename S, int D>
class Single {
 public:
  static constexpr int kComponents = 1;
  static constexpr int kDimensions = D;
  // Types:
  using Scalar = S;
  using Vector = algebra::Vector<Scalar, kDimensions>;
  using Jacobian = Scalar;
  using Coefficient = algebra::Vector<Jacobian, kDimensions>;
  using Conservative = Scalar;
  using Flux = Scalar;
  using Speed = Scalar;
  using MatKx1 = algebra::Matrix<Scalar, kComponents, 1>;
  // Constructor:
  Single() : a_const_(1) {}
  explicit Single(Jacobian const& a_const) : a_const_(a_const) {}
  // Get F on T Axia:
  Flux GetFluxUpwind(const Conservative& left, const Conservative& right)
      const {
    if (0 < a_const_) {
      return left * a_const_;
    } else {
      return right* a_const_;
    }
  }
  Flux GetFluxUpwind(const MatKx1& left, const MatKx1& right) const {
    return GetFluxUpwind(left[0], right[0]);
  }
  // Get F of U
  Flux GetFlux(const Conservative& state) const {
    return state * a_const_;
  }
  Flux GetFlux(const MatKx1& state) const {
    return GetFlux(state[0]);
  }

 private:
  Jacobian a_const_;
};

}  // namespace simple
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_SIMPLE_SINGLE_HPP_
