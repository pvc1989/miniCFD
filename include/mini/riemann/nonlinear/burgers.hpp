//  Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_RIEMANN_NONLINEAR_BURGERS_HPP_
#define MINI_RIEMANN_NONLINEAR_BURGERS_HPP_

#include <array>
#include <cmath>

#include "mini/algebra/column.hpp"

namespace mini {
namespace riemann {
namespace nonlinear {

template <int D>
class Burgers {
 public:
  static constexpr int kDim = D;
  // Types:
  using Scalar = double;
  using Vector = algebra::Column<double, kDim>;
  using Jacobi = double;
  using State = double;
  using Flux = double;
  using Coefficient = algebra::Column<Jacobi, kDim>;
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
  // Get F of U
  Flux GetFlux(State const& state) const {
    return state * state * k_ / 2;
  }

 private:
  Jacobi k_;
};

}  // namespace nonlinear
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_NONLINEAR_BURGERS_HPP_
