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
  Burgers() {}
  // Get F on T Axia
  Flux GetFluxOnTimeAxis(State u_l, State u_r) {
    SetInitial(u_l, u_r);
    DetermineWaveStructure();
    return GetFlux(GetState(/* x = */0.0, /* t = */1.0));
  }
  // Get F of U
  Flux GetFlux(State state) {
      return state * state / 2;
  }

 private:
  // Set U_l and U_r
  void SetInitial(State u_l, State u_r) {
    u_l_ = u_l;
    u_r_ = u_r;
  }
  void DetermineWaveStructure() {
    if (u_l_ >= u_r_) {
      double v = (u_l_ + u_r_) / 2;
      v_l_ = v;
      v_r_ = v;
    } else {
      v_l_ = u_l_;
      v_r_ = u_r_;
    }
  }
  // Get U at (x, t)
  State GetState(double x, double t) {
    double slope = x / t;
    if (slope <= v_l_) {
      return u_l_;
    }
    else if (slope >= v_r_) {
      return u_r_;
    }
    else if (v_l_ < slope < v_r_) {
      return slope;
    }
    return slope;
  }
  State u_l_, u_r_;
  State v_l_, v_r_;
};

}  //  namespace riemann
}  //  namespace mini

#endif  //  MINI_RIEMANN_BURGERS_HPP_
