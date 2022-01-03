// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_RIEMANN_EULER_AUSM_HPP_
#define MINI_RIEMANN_EULER_AUSM_HPP_

#include <array>
#include <algorithm>
#include <cmath>

#include "mini/riemann/euler/types.hpp"

namespace mini {
namespace riemann {
namespace euler {

template <class GasModel, int kDim>
class Ausm;

template <class GasModel>
class Ausm<GasModel, 1> {
 public:
  constexpr static int kDim = 1;
  // Types:
  using Gas = GasModel;
  using Scalar = typename Gas::Scalar;
  using Flux = FluxTuple<Scalar, 1>;
  using Conservative = ConservativeTuple<Scalar, 1>;
  using Primitive = PrimitiveTuple<Scalar, 1>;
  using State = Primitive;
  using Vector = typename State::Vector;
  using Speed = Scalar;
  // Get F on T Axia
  Flux GetFluxOnTimeAxis(State const& left, State const& right) {
    Flux flux_positive = GetPositiveFlux(left);
    Flux flux_negative = GetNegativeFlux(right);
    flux_positive += flux_negative;
    return flux_positive;
  }
  // Get F of U
  Flux GetFlux(State const& state) {
    auto rho_u = state.rho() * state.u();
    auto rho_u_u = rho_u * state.u();
    return {rho_u, rho_u_u + state.p(),
            state.u() * (state.p() * Gas::GammaOverGammaMinusOne()
                       + 0.5 * rho_u_u)};
  }

 private:
  Flux GetPositiveFlux(State const& state) {
    double p_positive   = state.p();
    double a = Gas::GetSpeedOfSound(state);
    double mach = state.u() / a;
    double mach_positive = mach;
    double h = a * a / Gas::GammaMinusOne() + state.u() * state.u() * 0.5;
    Flux flux = {1, state.u(), h};
    if (mach >= -1 && mach <= 1) {
      mach_positive = (mach + 1) * (mach + 1) * 0.25;
      p_positive = state.p() * (mach + 1) * 0.5;
    } else if (mach < -1) {
      mach_positive = 0.0;
      p_positive = 0.0;
    }
    double temp = state.rho() * a * mach_positive;
    flux *= temp;
    flux.momentumX() += p_positive;
    return flux;
  }
  Flux GetNegativeFlux(State state) {
    double p_negative = state.p();
    double a = Gas::GetSpeedOfSound(state);
    double mach = state.u() / a;
    double mach_negative = mach;
    double h = a * a / Gas::GammaMinusOne() + state.u() * state.u() * 0.5;
    Flux flux = {1, state.u(), h};
    if (mach >= -1 && mach <= 1) {
      mach_negative = - (mach - 1) * (mach - 1) * 0.25;
      p_negative = - state.p() * (mach - 1) * 0.5;
    } else if (mach > 1) {
      mach_negative = 0.0;
      p_negative = 0.0;
    }
    flux *= state.rho() * a * mach_negative;
    flux.momentumX() += p_negative;
    return flux;
  }
};

template <class GasModel>
class Ausm<GasModel, 2> {
 public:
  constexpr static int kDim = 2;
  // Types:
  using Gas = GasModel;
  using Scalar = typename Gas::Scalar;
  using Flux = FluxTuple<Scalar, 2>;
  using Conservative = ConservativeTuple<Scalar, 2>;
  using Primitive = PrimitiveTuple<Scalar, 2>;
  using State = Primitive;
  using Vector = typename State::Vector;
  using Speed = Scalar;
  // Get F on T Axia
  Flux GetFluxOnTimeAxis(State const& left, State const& right) {
    Flux flux_positive = GetPositiveFlux(left);
    Flux flux_negative = GetNegativeFlux(right);
    flux_positive += flux_negative;
    return flux_positive;
  }
  // Get F of U
  Flux GetFlux(State const& state) {
    auto rho_u = state.rho() * state.u();
    auto rho_v = state.rho() * state.v();
    auto rho_u_u = rho_u * state.u();
    return {rho_u, rho_u_u + state.p(), rho_v * state.u(),
            state.u() * (state.p() * Gas::GammaOverGammaMinusOne()
                       + 0.5 * (rho_u_u + rho_v * state.v()))};
  }

 private:
  Flux GetPositiveFlux(State const& state) {
    double p_positive = state.p();
    double a = Gas::GetSpeedOfSound(state);
    double mach = state.u() / a;
    double mach_positive = mach;
    double h = a * a / Gas::GammaMinusOne() + state.u() * state.u() * 0.5 +
                                              state.v() * state.v() * 0.5;
    Flux flux = {1, state.u(), state.v(), h};
    if (mach >= -1 && mach <= 1) {
      mach_positive = (mach + 1) * (mach + 1) * 0.25;
      p_positive = state.p() * (mach + 1) * 0.5;
    } else if (mach < -1) {
      mach_positive = 0.0;
      p_positive = 0.0;
    }
    double temp = state.rho() * a * mach_positive;
    flux *= temp;
    flux.momentumX() += p_positive;
    return flux;
  }
  Flux GetNegativeFlux(State state) {
    double p_negative = state.p();
    double a = Gas::GetSpeedOfSound(state);
    double mach = state.u() / a;
    double mach_negative = mach;
    double h = a * a / Gas::GammaMinusOne() + state.u() * state.u() * 0.5 +
                                              state.v() * state.v() * 0.5;
    Flux flux = {1, state.u(), state.v(), h};
    if (mach >= -1 && mach <= 1) {
      mach_negative = - (mach - 1) * (mach - 1) * 0.25;
      p_negative = - state.p() * (mach - 1) * 0.5;
    } else if (mach > 1) {
      mach_negative = 0.0;
      p_negative = 0.0;
    }
    double temp = state.rho() * a * mach_negative;
    flux *= temp;
    flux.momentumX() += p_negative;
    return flux;
  }
};

template <class GasModel>
class Ausm<GasModel, 3> {
 public:
  constexpr static int kDim = 3;
  // Types:
  using Gas = GasModel;
  using Scalar = typename Gas::Scalar;
  using Flux = FluxTuple<Scalar, 3>;
  using Conservative = ConservativeTuple<Scalar, 3>;
  using Primitive = PrimitiveTuple<Scalar, 3>;
  using State = Primitive;
  using Vector = typename State::Vector;
  using Speed = Scalar;
  // Get F on T Axia
  Flux GetFluxOnTimeAxis(State const& left, State const& right) {
    Flux flux_positive = GetPositiveFlux(left);
    Flux flux_negative = GetNegativeFlux(right);
    flux_positive += flux_negative;
    return flux_positive;
  }
  // Get F of U
  Flux GetFlux(State const& state) {
    auto rho_u = state.rho() * state.u();
    auto rho_v = state.rho() * state.v();
    auto rho_w = state.rho() * state.w();
    auto rho_u_u = rho_u * state.u();
    return {rho_u, rho_u_u + state.p(), rho_v * state.u(), rho_w * state.u(),
            state.u() * (state.p() * Gas::GammaOverGammaMinusOne()
            + 0.5 * (rho_u_u + rho_v * state.v() + rho_w * state.w()))};
  }

 private:
  Flux GetPositiveFlux(State const& state) {
    double p_positive = state.p();
    double a = Gas::GetSpeedOfSound(state);
    double mach = state.u() / a;
    double mach_positive = mach;
    double h = a * a / Gas::GammaMinusOne() + state.u() * state.u() * 0.5
              + state.v() * state.v() * 0.5 + state.w() * state.w() * 0.5;
    Flux flux = {1, state.u(), state.v(), state.w(), h};
    if (mach >= -1 && mach <= 1) {
      mach_positive = (mach + 1) * (mach + 1) * 0.25;
      p_positive = state.p() * (mach + 1) * 0.5;
    } else if (mach < -1) {
      mach_positive = 0.0;
      p_positive = 0.0;
    }
    double temp = state.rho() * a * mach_positive;
    flux *= temp;
    flux.momentumX() += p_positive;
    return flux;
  }
  Flux GetNegativeFlux(State state) {
    double p_negative = state.p();
    double a = Gas::GetSpeedOfSound(state);
    double mach = state.u() / a;
    double mach_negative = mach;
    double h = a * a / Gas::GammaMinusOne() + state.u() * state.u() * 0.5
              + state.v() * state.v() * 0.5 + state.w() * state.w() * 0.5;
    Flux flux = {1, state.u(), state.v(), state.w(), h};
    if (mach >= -1 && mach <= 1) {
      mach_negative = - (mach - 1) * (mach - 1) * 0.25;
      p_negative = - state.p() * (mach - 1) * 0.5;
    } else if (mach > 1) {
      mach_negative = 0.0;
      p_negative = 0.0;
    }
    double temp = state.rho() * a * mach_negative;
    flux *= temp;
    flux.momentumX() += p_negative;
    return flux;
  }
};

}  //  namespace euler
}  //  namespace riemann
}  //  namespace mini

#endif  //  MINI_RIEMANN_EULER_AUSM_HPP_
