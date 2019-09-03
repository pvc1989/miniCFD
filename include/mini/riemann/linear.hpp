//  Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_RIEMANN_LINEAR_HPP_
#define MINI_RIEMANN_LINEAR_HPP_

#include <cmath>
#include <vector>

namespace mini {
namespace riemann {

template <class State>
class Flux {
 public:
  Flux GetFlux(State state);
};

template <class State>
class Problem {
 public:
  Flux<State> FluxOnTimeAxis(State state_left, State state_right);
};

class SingleWave {
 public:
  // Types:
  using State = double;
  using Flux = double;
  using Speed = double;
  // Constructor:
  explicit SingleWave(Speed a_const) : a_const_(a_const) {}
  // Get F on T Axia
  Flux GetFluxOnTimeAxis(State u_l, State u_r) {
    SetInitial(u_l, u_r);
    return GetFlux(GetState(/* x = */0.0, /* t = */1.0));
  }
  // Get F of U
  Flux GetFlux(State state) { return state * a_const_; }

 private:
  // Set U_l and U_r
  void SetInitial(State u_l, State u_r) {
    u_l_ = u_l;
    u_r_ = u_r;
  }
  // Get U at (x, t)
  double GetState(double x, double t) {
    if (t <= 0) {
      if (x <= 0) {
        return u_l_;
      } else {
        return u_r_;
      }
    } else {
      if (x / t <= a_const_) {
        return u_l_;
      } else {
        return u_r_;
      }
    }
  }

 private:
  State u_l_, u_r_;
  Speed a_const_;
};

}  //  namespace riemann
}  //  namespace mini

#endif  //  MINI_RIEMANN_LINEAR_HPP_
