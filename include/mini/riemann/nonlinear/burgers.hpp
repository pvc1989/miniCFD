//  Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_RIEMANN_NONLINEAR_BURGERS_HPP_
#define MINI_RIEMANN_NONLINEAR_BURGERS_HPP_

#include <array>
#include <cmath>

#include "mini/algebra/eigen.hpp"

namespace mini {
namespace riemann {
namespace nonlinear {

template <int D>
class Burgers {
 public:
  static constexpr int kDim = D;
  static constexpr int kFunc = 1;
  // Types:
  using Scalar = double;
  using Vector = algebra::Vector<double, kDim>;
  using Jacobi = double;
  using State = double;
  using Flux = double;
  using Coefficient = algebra::Vector<Jacobi, kDim>;
  using MatKx1 = algebra::Matrix<Scalar, kFunc, 1>;
  // Constructor:
  Burgers() : k_(1) {}
  explicit Burgers(double k) : k_(k) {}
  // Get F on T Axia
  Flux GetFluxOnTimeAxis(State const& left, State const& right) const {
    if (k_ == 0.0) { return 0.0; }
    State left_ = k_ * left;
    State right_ = k_ * right;
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
  Flux GetFluxOnTimeAxis(const MatKx1& left, const MatKx1& right) const {
    return GetFluxOnTimeAxis(left[0], right[0]);
  }
  // Get F of U
  Flux GetFlux(State const& state) const {
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
