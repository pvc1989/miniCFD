//  Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_RIEMANN_BURGERS_HPP_
#define MINI_RIEMANN_BURGERS_HPP_

#include <cmath>
#include <array>

namespace mini {
namespace riemann {

class Burgers {
 public:
  // Types:
  using Jacobi = double;
  using State = double;
  using Flux = double;
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
  double k_;
};

}  //  namespace riemann
}  //  namespace mini

#endif  //  MINI_RIEMANN_BURGERS_HPP_
