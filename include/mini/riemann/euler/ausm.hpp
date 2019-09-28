#ifndef MINI_RIEMANN_EULER_AUSM_HPP_
#define MINI_RIEMANN_EULER_AUSM_HPP_

#include <array>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace mini {
namespace riemann {
namespace euler{

template <class Gas>
class Ausm {
 public:
  // Types:
  using State = typename Gas::State;
  using Flux = std::array<double, 3>;
  // Get F on T Axia
  Flux GetFluxOnTimeAxis(State const& left, State const& right) {
    Flux flux_positive = GetPositiveFlux(left);
    Flux flux_negative = GetNegativeFlux(right);
    Flux flux;
    for (int i = 0; i < 3; i++) {
      flux[i] = flux_positive[i] + flux_negative[i];
    }
    return flux;
  }
  // Get F of U
  Flux GetFlux(State const& state) {
    Flux flux;
    double e = state.p / Gas::GammaMinusOne() +
               state.rho * state.u * state.u / 2;
    flux[0] = state.rho * state.u;
    flux[1] = flux[0] * state.u + state.p;
    flux[2] = state.u * (e + state.p);
    return flux;
  }
  double GetTangentialComponent(double left, double right) {
    return left * left_tangential_weight_ + right * right_tangential_weight_;
  }
 private:
  double  left_tangential_weight_;
  double  right_tangential_weight_;
  Flux GetPositiveFlux(State const& state) {
    double p_positive   = state.p;
    double a = Gas::GetSpeedOfSound(state);
    double mach = state.u / a;
    double mach_positive = mach;
    double h = a * a / Gas::GammaMinusOne() + state.u * state.u / 2;
    left_tangential_weight_ = mach_positive * state.rho * a;
    Flux flux = {1, state.u, h};
    if (mach >= -1 && mach <= 1) {
      mach_positive = (mach + 1) * (mach + 1) / 4;
      p_positive = state.p * (mach + 1) / 2;
    }
    else if (mach < -1) {
      mach_positive = 0.0;
      p_positive = 0.0;
    }
    double temp = state.rho * a * mach_positive;;
    for (int i = 0; i < 3; i++) {
      flux[i] *= temp;
    }
    flux[1] += p_positive;
    return flux;
  }
  Flux GetNegativeFlux(State state) {
    double p_negative = state.p;
    double a = Gas::GetSpeedOfSound(state);
    double mach = state.u / a;
    double mach_negative = mach;
    double h = a * a / Gas::GammaMinusOne() + state.u * state.u / 2;
    right_tangential_weight_ = mach_negative * state.rho * a;
    Flux flux = {1, state.u, h};
    if (mach >= -1 && mach <= 1) {
      mach_negative = - (mach - 1) * (mach - 1) / 4;
      p_negative = - state.p * (mach - 1) / 2;
    }
    else if (mach > 1) {
      mach_negative = 0.0;
      p_negative = 0.0;
    }
    for (int i = 0; i < 3; i++) {
      flux[i] *= state.rho * a * mach_negative;
    }
    flux[1] += p_negative;
    return flux;
  }
};

}  //  namespace euler
}  //  namespace riemann
}  //  namespace mini

#endif  //  MINI_RIEMANN_EULER_AUSM_HPP_
