// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_RIEMANN_EULER_HLLC_HPP_
#define MINI_RIEMANN_EULER_HLLC_HPP_

#include <array>
#include <algorithm>
#include <cmath>

#include "mini/riemann/euler/types.hpp"

namespace mini {
namespace riemann {
namespace euler {

template <class GasModel, int kDim = 1>
class Hllc;
template <class GasModel>
class Hllc<GasModel, 1> {
 public:
  // Types:
  using Gas = GasModel;
  using Conservative = Conservative<1>;
  using Primitive = Primitive<1>;
  using State = Primitive;
  using Flux = Flux<1>;
  using Scalar = typename State::Scalar;
  using Vector = typename State::Vector;
  using Speed = Scalar;
  // Get F on T Axia
  Flux GetFluxOnTimeAxis(State const& left, State const& right) {
    Initialize(left, right);
    Flux flux;
    if (0.0 <= wave_left_) {
      flux = GetFlux(left);
    } else if (wave_right_ <= 0.0) {
      flux = GetFlux(right);
    } else if (0.0 <= wave_star_) {
      flux = GetStarFlux(left, wave_left_);
    } else if (wave_star_ < 0.0) {
      flux = GetStarFlux(right, wave_right_);
    }
    return flux;
  }
  // Get F of U
  Flux GetFlux(const State& state) {
    auto rho_u = state.rho() * state.u();
    auto rho_u_u = rho_u * state.u();
    return {rho_u, rho_u_u + state.p(),
            state.u() * (state.p() * Gas::GammaOverGammaMinusOne()
                       + 0.5 * rho_u_u)};
  }

 private:
  Speed wave_star_;
  Speed wave_left_;
  Speed wave_right_;
  void Initialize(State const& left, State const& right) {
    double rho_average = (left.rho() + right.rho()) / 2;
    double a_left = Gas::GetSpeedOfSound(left);
    double a_right = Gas::GetSpeedOfSound(right);
    double a_average = (a_left + a_right) / 2;
    double p_pvrs = (left.p() + right.p() - (right.u() - left.u()) *
                     rho_average * a_average) / 2;
    double p_estimate = std::max(0.0, p_pvrs);
    wave_left_ = left.u() - a_left * GetQ(p_estimate, left.p());
    wave_right_ = right.u() + a_right * GetQ(p_estimate, right.p());
    wave_star_ = (right.p() - left.p() +
                   left.rho() *  left.u() *  (wave_left_ -  left.u()) -
                  right.rho() * right.u() * (wave_right_ - right.u())) /
                  (left.rho() * (wave_left_  -  left.u()) -
                  right.rho() * (wave_right_ - right.u()));
  }
  double GetQ(Scalar const& p_estimate, Scalar const& p_k) {
    if (p_estimate <= p_k) {
      return 1.0;
    } else {
      double temp = 1 + Gas::GammaPlusOneOverTwo() * (p_estimate / p_k - 1) /
                    Gas::Gamma();
      return std::sqrt(temp);
    }
  }
  Flux GetStarFlux(State const& state, Speed const& wave_k) {
    Flux flux = GetFlux(state);
    auto energy = state.p() / Gas::GammaMinusOne() +
                  state.rho() * (state.u() * state.u()) * 0.5;
    double temp = state.rho() * (wave_k - state.u()) / (wave_k - wave_star_);
    Conservative u_k = Gas::PrimitiveToConservative(state);
    Conservative u_star_k;
    u_star_k.mass = temp;
    u_star_k.momentum[0] = wave_star_ * temp;
    u_star_k.energy = energy / state.rho() + (wave_star_ - state.u()) *
                (wave_star_ + state.p() / (state.rho() * (wave_k - state.u())));
    u_star_k.energy *= temp;
    u_star_k -= u_k;
    u_star_k *= wave_k;
    flux += u_star_k;
    return flux;
  }
};

template <class GasModel>
class Hllc<GasModel, 2> {
 public:
  // Types:
  using Gas = GasModel;
  using Conservative = Conservative<2>;
  using Primitive = Primitive<2>;
  using State = Primitive;
  using Flux = Flux<2>;
  using Scalar = typename State::Scalar;
  using Vector = typename State::Vector;
  using Speed = Scalar;
  // Get F on T Axia
  Flux GetFluxOnTimeAxis(State const& left, State const& right) {
    Initialize(left, right);
    Flux flux;
    if (0.0 <= wave_left_) {
      flux = GetFlux(left);
    } else if (wave_right_ <= 0.0) {
      flux = GetFlux(right);
    } else if (0.0 <= wave_star_) {
      flux = GetStarFlux(left, wave_left_);
    } else if (wave_star_ < 0.0) {
      flux = GetStarFlux(right, wave_right_);
    }
    return flux;
  }
  // Get F of U
  Flux GetFlux(const State& state) {
    auto rho_u = state.rho() * state.u();
    auto rho_v = state.rho() * state.v();
    auto rho_u_u = rho_u * state.u();
    return {rho_u, rho_u_u + state.p(), rho_v * state.u(),
            state.u() * (state.p() * Gas::GammaOverGammaMinusOne()
                       + 0.5 * (rho_u_u + rho_v * state.v()))};
  }

 private:
  Speed wave_star_;
  Speed wave_left_;
  Speed wave_right_;
  // double rho_u_time_axis_;
  void Initialize(State const& left, State const& right) {
    double rho_average = (left.rho() + right.rho()) / 2;
    double a_left = Gas::GetSpeedOfSound(left);
    double a_right = Gas::GetSpeedOfSound(right);
    double a_average = (a_left + a_right) / 2;
    double p_pvrs = (left.p() + right.p() - (right.u() - left.u()) *
                     rho_average * a_average) / 2;
    double p_estimate = std::max(0.0, p_pvrs);
    wave_left_ = left.u() - a_left * GetQ(p_estimate, left.p());
    wave_right_ = right.u() + a_right * GetQ(p_estimate, right.p());
    wave_star_ = (right.p() - left.p() +
                   left.rho() *  left.u() *  (wave_left_ -  left.u()) -
                  right.rho() * right.u() * (wave_right_ - right.u())) /
                  (left.rho() * (wave_left_  -  left.u()) -
                  right.rho() * (wave_right_ - right.u()));
  }
  double GetQ(Scalar const& p_estimate, Scalar const& p_k) {
    if (p_estimate <= p_k) {
      return 1.0;
    } else {
      double temp = 1 + Gas::GammaPlusOneOverTwo() * (p_estimate / p_k - 1) /
                    Gas::Gamma();
      return std::sqrt(temp);
    }
  }
  Flux GetStarFlux(State const& state, Speed const& wave_k) {
    Flux flux = GetFlux(state);
    double energy = state.p() / Gas::GammaMinusOne() +
              state.rho() * (state.u() * state.u() +
                             state.v() * state.v()) * 0.5;
    double temp = state.rho() * (wave_k - state.u()) / (wave_k - wave_star_);
    Conservative u_k = Gas::PrimitiveToConservative(state);
    Conservative u_star_k;
    u_star_k.mass = temp;
    u_star_k.momentum[0] = wave_star_ * temp;
    u_star_k.momentum[1] = state.v() * temp;
    u_star_k.energy = energy / state.rho() + (wave_star_ - state.u()) *
                (wave_star_ + state.p() / (state.rho() * (wave_k - state.u())));
    u_star_k.energy *= temp;
    u_star_k -= u_k;
    u_star_k *= wave_k;
    flux += u_star_k;
    return flux;
  }
};

}  //  namespace euler
}  //  namespace riemann
}  //  namespace mini

#endif  //  MINI_RIEMANN_EULER_HLLC_HPP_
