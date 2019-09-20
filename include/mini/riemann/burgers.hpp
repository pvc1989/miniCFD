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
  using State = double;
  using Flux = double;
  // Constructor:
  explicit Burgers(double k) : k_(k) {}
  // Get F on T Axia
  Flux GetFluxOnTimeAxis(State u_l, State u_r) {
    if (k_ == 0.0) { return 0.0; }
    SetInitial(u_l, u_r);
    DetermineWaveStructure();
    return GetFlux(GetState(/* slope */0.0));
  }
  // Get F of U
  Flux GetFlux(State state) const {
    return state * state * k_ / 2.0;
  }

 private:
  using Slope = double;
  // Set U_l and U_r
  void SetInitial(State u_l, State u_r) {
    u_l_ = u_l;
    u_r_ = u_r;
  }
  void DetermineWaveStructure() {
    a_l_ = k_ * u_l_;
    a_r_ = k_ * u_r_;
    if (a_l_ >= a_r_) {  // shock
      a_l_ = (a_l_ + a_r_) / 2;
      a_r_ = a_l_;
    } else {  // expansion
      // a_l_, a_r_ already calculated.
    }
  }
  // Get U on {(x, t) : x = slope * t}
  State GetState(Slope slope) {
    if (slope <= a_l_) {
      return u_l_;
    } else if (slope >= a_r_) {
      return u_r_;
    } else {  // a_l_ < slope < a_r_
      return slope / k_;
    }
  }
  double k_;
  State u_l_, u_r_;
  Slope a_l_, a_r_;
};

}  //  namespace riemann
}  //  namespace mini

#endif  //  MINI_RIEMANN_BURGERS_HPP_
