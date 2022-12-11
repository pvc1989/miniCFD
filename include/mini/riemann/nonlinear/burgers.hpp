//  Copyright 2019 PEI Weicheng and YANG Minghao

#ifndef MINI_RIEMANN_NONLINEAR_BURGERS_HPP_
#define MINI_RIEMANN_NONLINEAR_BURGERS_HPP_

#include <array>
#include <cmath>

#include "mini/algebra/eigen.hpp"

namespace mini {
namespace riemann {
namespace nonlinear {

template <typename S, int D>
class Burgers {
 public:
  static constexpr int kDimensions = D;
  static constexpr int kComponents = 1;
  // Types:
  using Scalar = S;
  using Vector = algebra::Vector<Scalar, kDimensions>;
  using Jacobi = Scalar;
  using Conservative = Scalar;
  using Flux = Scalar;
  using Coefficient = algebra::Vector<Jacobi, kDimensions>;
  using MatKx1 = algebra::Matrix<Scalar, kComponents, 1>;
  // Constructor:
  Burgers() : k_(1) {}
  explicit Burgers(Scalar k) : k_(k) {}
  // Get F on T Axia
  Flux GetFluxUpwind(Conservative const& left, Conservative const& right)
      const {
    if (k_ == 0.0) { return 0.0; }
    Conservative left_ = k_ * left;
    Conservative right_ = k_ * right;
    if (left_ >= right_) {  // shock
      left_ = (left_ + right_) / 2;
      right_ = left_;
    }
    if (0 <= left_) {
      return GetFlux(left);
    } else if (0 >= right_) {
      return GetFlux(right);
    } else {  // left_ < slope < right_
      return GetFlux(0 / k_);
    }
  }
  Flux GetFluxUpwind(const MatKx1& left, const MatKx1& right) const {
    return GetFluxUpwind(left[0], right[0]);
  }
  // Get F of U
  Flux GetFlux(Conservative const& state) const {
    return state * state * k_ / 2;
  }
  Flux GetFlux(const MatKx1& state) const {
    return GetFlux(state[0]);
  }

 private:
  Jacobi k_;
};

}  // namespace nonlinear
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_NONLINEAR_BURGERS_HPP_
