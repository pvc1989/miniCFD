#ifndef MINI_RIEMANN_HLLC_HPP_
#define MINI_RIEMANN_HLLC_HPP_

#include <array>
#include <algorithm>
#include <cmath>
#include <iostream>

#include <iostream>

namespace mini {
namespace riemann {
namespace euler{

template <class Gas>
class Hllc {
 public:
  // Types:
  using State = typename Gas::State;
  using Flux = std::array<double, 3>;
  using Column = Flux;
  // Get F on T Axia
  Flux GetFluxOnTimeAxis(const State& left, const State& right) {
    Initialize(left, right);
    Flux flux;
    if (0.0 <= wave_left_) {
      flux = GetFlux(left);
    }
    else if (wave_right_ <= 0.0) {
      flux = GetFlux(right);
    }
    else if (0.0 <= wave_star_) {
      flux = GetStarFlux(left, wave_left_);
    }
    else if (wave_star_ < 0.0) {
      flux = GetStarFlux(right, wave_right_);
    }
    rho_u_time_axis_ = flux[0];
    return flux;
  }
  // Get F of U
  Flux GetFlux(const State& state) {
    Flux flux;
    energy_ = state.p / Gas::GammaMinusOne() +
              state.rho * (state.u * state.u) / 2;
    flux[0] = state.rho * state.u;
    flux[1] = flux[0] * state.u + state.p;
    flux[2] = state.u * (energy_ + state.p);
    return flux;
  }
  // Get position of wave_star
  double GetTangentialComponent(double left, double right) {
    if (wave_star_ >= 0.0) {
      return rho_u_time_axis_ * left;
    } else {
      return rho_u_time_axis_ * right;
    }
  }

 private:
  double wave_star_;
  double wave_left_;
  double wave_right_;
  double energy_;
  double rho_u_time_axis_;
  void Initialize(const State& left, const State& right) {
    double rho_average = (left.rho + right.rho) / 2;
    double a_left = Gas::GetSpeedOfSound(left);
    double a_right = Gas::GetSpeedOfSound(right);
    double a_average = (a_left + a_right) / 2;
    double p_pvrs = (left.p + right.p - (right.u - left.u) *
                     rho_average * a_average) / 2;
    double p_estimate = std::max(0.0, p_pvrs);
    wave_left_ = left.u - a_left * GetQ(p_estimate ,left.p);
    wave_right_ = right.u + a_right * GetQ(p_estimate ,right.p);
    wave_star_ = (right.p - left.p + 
                   left.rho *  left.u * ( wave_left_ -  left.u) -
                  right.rho * right.u * (wave_right_ - right.u)) /
                  (left.rho * ( wave_left_ -  left.u) - 
                  right.rho * (wave_right_ - right.u));
  }
  double GetQ(const double& p_estimate, const double& p_k) {
    if (p_estimate <= p_k) {
      return 1.0;
    } else {
      double temp = 1 + Gas::GammaPlusOneOverTwo() * (p_estimate / p_k - 1) /
                    Gas::Gamma();
      return std::sqrt(temp);
    }
  }
  Flux GetStarFlux(const State& state, const double& wave_k) {
    Flux flux = GetFlux(state);
    double temp = state.rho * (wave_k - state.u) / (wave_k - wave_star_);
    Column u_k = {state.rho, state.rho * state.u, energy_};
    Column f_change;
    f_change[0] = temp;
    f_change[1] = wave_star_ * temp;
    f_change[2] = energy_ / state.rho + (wave_star_ - state.u) * 
                (wave_star_ + state.p / (state.rho * (wave_k - state.u)));
    f_change[2] *= temp;
    for (int i = 0; i < 3; i++) {
      flux[i] += (f_change[i] - u_k[i]) * wave_k;
    }
    return flux;
  }
};

}  //  namespace
}  //  namespace riemann
}  //  namespace mini

#endif  //  MINI_RIEMANN_HLLC_HPP_
