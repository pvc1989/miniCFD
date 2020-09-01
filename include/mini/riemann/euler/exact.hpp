//  Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_RIEMANN_EULER_EXACT_HPP_
#define MINI_RIEMANN_EULER_EXACT_HPP_

#include <cassert>
#include <cmath>

#include "mini/riemann/euler/types.hpp"

namespace mini {
namespace riemann {
namespace euler {

template <int kField>
constexpr double AddOrMinus(double x, double y);
template <>
constexpr double AddOrMinus<1>(double x, double y) { return x + y; }
template <>
constexpr double AddOrMinus<3>(double x, double y) { return x - y; }
template <int kField>
bool TimeAxisAfterExpansion(double u, double a);
template <>
bool TimeAxisAfterExpansion<1>(double u, double a) { return u - a < 0; }
template <>
bool TimeAxisAfterExpansion<3>(double u, double a) { return u + a > 0; }
template <int kField>
bool TimeAxisBeforeExpansion(double u, double a);
template <>
bool TimeAxisBeforeExpansion<1>(double u, double a) { return u - a > 0; }
template <>
bool TimeAxisBeforeExpansion<3>(double u, double a) { return u + a < 0; }

template <class Gas, int kDim>
class Implementor {
 public:
  // Types:
  using FluxType = Flux<kDim>;
  using ConservativeType = Conservative<kDim>;
  using PrimitiveType = Primitive<kDim>;
  using State = PrimitiveType;
  using Scalar = typename State::Scalar;
  using Vector = typename State::Vector;
  using Speed = Scalar;
  // Data:
  Speed star_u{0.0};
  // Get U on t-Axis
  State GetStateOnTimeAxis(State const& left, State const& right) {
    // Construct the function of speed change, aka the pressure function.
    auto u_change_given = right.u() - left.u();
    auto u_change__left = SpeedChange(left);
    auto u_change_right = SpeedChange(right);
    auto f = [&](double p) {
      return u_change__left(p) + u_change_right(p) + u_change_given;
    };
    auto f_prime = [&](double p) {
      return u_change__left.Prime(p) + u_change_right.Prime(p);
    };
    if (f(0) < 0) {  // Ordinary case: Wave[2] is a contact.
      auto star = State{0, 0, 0};
      star.p() = FindRoot(f, f_prime, left.p());
      star.u() = 0.5 * (right.u() + u_change_right(star.p())
                        +left.u() - u_change__left(star.p()));
      star_u = star.u();
      if (0 < star.u()) {  // Axis[t] <<< Wave[2]
        if (star.p() >= left.p()) {  // Wave[1] is a shock.
          return GetStateNearShock<1>(left, &star);
        } else {  // star.p() < left.p() : Wave[1] is an expansion.
          return GetStateNearExpansion<1>(left, &star);
        }
      } else {  // star.u() < 0 : Wave[2] <<< Axis[t] <<< Wave[3].
        if (star.p() >= right.p()) {  // Wave[3] is a shock.
          return GetStateNearShock<3>(right, &star);
        } else {  // star.p() < right.p() : Wave[3] is an expansion.
          return GetStateNearExpansion<3>(right, &star);
        }
      }
    } else {  // The region BETWEEN Wave[1] and Wave[3] is vacuumed.
      return GetStateNearVacuum(left, right);
    }
  }

 private:
  // Helper method and class for the star region:
  template <class F, class Fprime>
  static double FindRoot(F&& f, Fprime&& f_prime, double x, double eps = 1e-8) {
    while (f(x) > 0) {
      x *= 0.5;
    }
    while (f(x) < -eps) {
      auto divisor = f_prime(x);
      assert(divisor != 0);
      x -= f(x) / divisor;
    }
    assert(std::abs(f(x)) < eps);
    return x;
  }
  class SpeedChange {
   public:
    explicit SpeedChange(State const& before)
        : rho_before_(before.rho()), p_before_(before.p()),
          a_before_(before.rho() == 0 ? 0 : Gas::GetSpeedOfSound(before)),
          p_const_(before.p() * Gas::GammaMinusOneOverTwo()) {
      assert(before.rho() >= 0 && before.p() >= 0);
    }
    double operator()(double p_after) const {
      double value;
      assert(p_after >= 0);
      if (p_after >= p_before_) {  // shock
        auto divisor = std::sqrt(rho_before_ * P(p_after));
        assert(divisor != 0);
        value = (p_after - p_before_) / divisor;
      } else {  // expansion
        constexpr auto exp = Gas::GammaMinusOneOverTwo() / Gas::Gamma();
        value = std::pow(p_after / p_before_, exp) - 1;
        value *= Gas::GammaMinusOneUnderTwo() * a_before_;
      }
      return value;
    }
    double Prime(double p_after) const {
      assert(p_after >= 0);
      double value;
      if (p_after >= p_before_) {  // shock
        auto p = P(p_after);
        value = Gas::GammaPlusOneOverFour() * (p_before_ - p_after);
        auto divisor = std::sqrt(rho_before_ * p);
        assert(divisor != 0);
        value = (1 + value / p) / divisor;
      } else {  // expansion
        constexpr double exp = -Gas::GammaPlusOneOverTwo() / Gas::Gamma();
        auto divisor = rho_before_ * a_before_;
        assert(divisor != 0);
        value = std::pow(p_after / p_before_, exp) / divisor;
      }
      return value;
    }

   private:
    double P(double p_after) const {
      return Gas::GammaPlusOneOverTwo() * p_after + p_const_;
    }

   private:
    double rho_before_, p_before_, a_before_, p_const_;
  };
  // Shock and related methods:
  template <int kField>
  class Shock {
   public:
    double u;
    Shock(State const& before, State const& after) : u(before.u()) {
      auto divisor = (after.u() - before.u()) * before.rho();
      u += (divisor ? (after.p() - before.p()) / divisor
                    : before.u());
    }
    double GetDensityAfterIt(State const& before, State const& after) const {
      auto divisor = after.u() - u;
      return divisor ? before.rho() * (before.u() - u) / divisor
                     : before.rho();
    }
  };
  static bool TimeAxisAfterShock(Shock<1> const& wave) { return wave.u < 0; }
  static bool TimeAxisAfterShock(Shock<3> const& wave) { return wave.u > 0; }
  template <int kField>
  static State GetStateNearShock(State const& before, State* after) {
    static_assert(kField == 1 || kField == 3);
    auto shock = Shock<kField>(before, *after);
    if (TimeAxisAfterShock(shock)) {  // i.e. (x=0, t) is AFTER the shock.
      after->rho() = before.rho();
      auto divisor = after->u() - shock.u;
      assert(divisor != 0);
      after->rho() *= (before.u() - shock.u) / divisor;
      return *after;
    } else {
      return before;
    }
  }
  // Expansion and related methods:
  template <int kField>
  class Expansion {
   public:
    double a_before, a_after;  // speed of sound before/after the wave
    double gri_1, gri_2;  // Generalized Riemann Invariants
    Expansion(State const& before, State const& after)
        : a_before(Gas::GetSpeedOfSound(before)),
          gri_1(before.p() / (std::pow(before.rho(), Gas::Gamma()))) {
      gri_2 = AddOrMinus<kField>(
        before.u(), a_before * Gas::GammaMinusOneUnderTwo());
      a_after = AddOrMinus<kField>(
        a_before, (before.u() - after.u()) * Gas::GammaMinusOneOverTwo());
    }
  };
  static State GetStateInsideExpansion(double gri_1, double gri_2) {
    constexpr auto r = Gas::GammaMinusOne() / Gas::GammaPlusOne();
    auto a = r * gri_2;
    auto a_square = a * a;
    auto rho = std::pow(a_square / gri_1 * Gas::OneOverGamma(),
                        Gas::OneOverGammaMinusOne());
    return {rho, a, a_square * rho * Gas::OneOverGamma()};
  }
  template <int kField>
  static State GetStateNearExpansion(State const& before, State* after) {
    static_assert(kField == 1 || kField == 3);
    auto wave = Expansion<kField>(before, *after);
    if (TimeAxisAfterExpansion<kField>(after->u(), wave.a_after)) {
      auto divisor = wave.a_after * wave.a_after;
      assert(divisor != 0);
      after->rho() = Gas::Gamma() * after->p() / divisor;
      return *after;
    } else if (TimeAxisBeforeExpansion<kField>(before.u(), wave.a_before)) {
      return before;
    } else {  // Axis[t] is inside the Expansion.
      return GetStateInsideExpansion(wave.gri_1, wave.gri_2);
    }
  }
  static State GetStateNearVacuum(State const& left, State const& right) {
    auto left_a = Gas::GetSpeedOfSound(left);
    if (left.u() > left_a) {  // Axis[t] <<< Wave[1].
      return left;
    }
    auto right_a = Gas::GetSpeedOfSound(right);
    if (right.u() + right_a < 0) {  // Wave[3] <<< Axis[t].
      return right;
    } else {  // Wave[1] <<< Axis[t] <<< Wave[3].
      auto gri_1 = left.p() / (std::pow(left.rho(), Gas::Gamma()));
      auto gri_2 = left.u() + left_a * Gas::GammaMinusOneUnderTwo();
      if (gri_2 >= 0) {  // Axis[t] is inside Wave[1].
        return GetStateInsideExpansion(gri_1, gri_2);
      } else {  // gri_2 < 0
        // Axis[t] is to the RIGHT of Wave[1].
        gri_1 = right.p() / (std::pow(right.rho(), Gas::Gamma()));
        gri_2 = right.u() - right_a * Gas::GammaMinusOneUnderTwo();
        if (gri_2 < 0) {  // Axis[t] is inside Wave[3].
          return GetStateInsideExpansion(gri_1, gri_2);
        } else {  // Axis[t] is inside the vacuumed region.
          return {0, 0, 0};
        }
      }
    }
  }
};

template <class GasModel, int kDim = 1>
class Exact;

template <class GasModel>
class Exact<GasModel, 1> : public Implementor<GasModel, 1> {
  using Base = Implementor<GasModel, 1>;

 public:
  // Types:
  using Gas = GasModel;
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using FluxType = typename Base::FluxType;
  using ConservativeType = typename Base::ConservativeType;
  using PrimitiveType = typename Base::PrimitiveType;
  using State = PrimitiveType;
  // Get F from U
  static FluxType GetFlux(State const& state) {
    auto rho_u = state.rho() * state.u();
    auto rho_u_u = rho_u * state.u();
    return {rho_u, rho_u_u + state.p(),
            state.u() * (state.p() * Gas::GammaOverGammaMinusOne()
                         + 0.5 * rho_u_u)};
  }
  // Get F on t-Axis
  FluxType GetFluxOnTimeAxis(State const& left, State const& right) {
    return GetFlux(GetStateOnTimeAxis(left, right));
  }
  // Get U on t-Axis
  State GetStateOnTimeAxis(State const& left, State const& right) {
    return Base::GetStateOnTimeAxis(left, right);
  }
};
template <class GasModel>
class Exact<GasModel, 2> : public Implementor<GasModel, 2> {
  using Base = Implementor<GasModel, 2>;

 public:
  // Types:
  using Gas = GasModel;
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using FluxType = typename Base::FluxType;
  using ConservativeType = typename Base::ConservativeType;
  using PrimitiveType = typename Base::PrimitiveType;
  using State = PrimitiveType;
  // Get F from U
  static FluxType GetFlux(State const& state) {
    auto rho_u = state.rho() * state.u();
    auto rho_v = state.rho() * state.v();
    auto rho_u_u = rho_u * state.u();
    return {rho_u, rho_u_u + state.p(), rho_v * state.u(),
            state.u() * (state.p() * Gas::GammaOverGammaMinusOne()
                         + 0.5 * (rho_u_u + rho_v * state.v()))};
  }
  // Get F on t-Axis
  FluxType GetFluxOnTimeAxis(State const& left, State const& right) {
    return GetFlux(GetStateOnTimeAxis(left, right));
  }
  // Get U on t-Axis
  State GetStateOnTimeAxis(State const& left, State const& right) {
    auto state = Base::GetStateOnTimeAxis(left, right);
    state.v() = this->star_u > 0 ? left.v() : right.v();
    return state;
  }
};

}  // namespace euler
}  // namespace riemann
}  // namespace mini

#endif  //  MINI_RIEMANN_EULER_EXACT_HPP_
