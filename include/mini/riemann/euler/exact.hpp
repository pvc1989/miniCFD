//  Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_RIEMANN_EULER_EXACT_HPP_
#define MINI_RIEMANN_EULER_EXACT_HPP_

#include <cmath>
#include <array>

namespace mini {
namespace riemann {
namespace euler {

template <class Gas>
class Exact {
 public:
  // Types:
  using State = typename Gas::State;
  using Flux = std::array<double, 3>;
  // Get F from U
  static Flux GetFlux(State const& state) {
    auto rho_u = state.rho * state.u;
    auto rho_u_u = rho_u * state.u;
    return {rho_u, rho_u_u + state.p,
            state.u * (state.p * Gas::GammaOverGammaMinusOne()
                       + 0.5 * rho_u_u)};
  }
  // Get U on t-Axis
  State GetStateOnTimeAxis(State const& left, State const& right) {
    // Construct the function of speed change, aka the pressure function.
    auto u_change_given = right.u - left.u;
    auto u_change__left = SpeedChange(left);
    auto u_change_right = SpeedChange(right);
    auto f = [&](double p) {
      return u_change__left(p) + u_change_right(p) + u_change_given;
    };
    auto f_prime = [&](double p) {
      return u_change__left.Prime(p) + u_change_right.Prime(p);
    };
    if (f(0) < 0) {  // Ordinary case: Wave[2] is a contact.
      auto star = State{0, 0, FindRoot(f, f_prime, left.p)};
      star.u = 0.5 * (right.u + u_change_right(star.p)
                      +left.u - u_change__left(star.p));
      if (0 < star.u) {  // Axis[t] <<< Wave[2]
        if (star.p >= left.p) {  // Wave[1] is a shock.
          return GetStateNearShock<1>(left, &star);
        } else {  // star.p < left.p : Wave[1] is an expansion.
          return GetStateNearExpansion<1>(left, &star);
        }
      } else {  // star.u < 0 : Wave[2] <<< Axis[t] <<< Wave[3].
        if (star.p >= right.p) {  // Wave[3] is a shock.
          return GetStateNearShock<3>(right, &star);
        } else {  // star.p < right.p : Wave[3] is an expansion.
          return GetStateNearExpansion<3>(right, &star);
        }
      }
    } else {  // The region BETWEEN Wave[1] and Wave[3] is vaccumed.
      return GetStateNearVaccum(left, right);
    }
  }
  // Get F on t-Axis
  Flux GetFluxOnTimeAxis(State const& left, State const& right) {
    return GetFlux(GetStateOnTimeAxis(left, right));
  }

 private:
  // Helper method and class for the star region:
  template <class F, class Fprime>
  static double FindRoot(F&& f, Fprime&& f_prime, double x, double eps = 1e-8) {
    while (f(x) > 0) {
      x *= 0.5;
    }
    while (f(x) < -eps) {
      x -= f(x) / f_prime(x);
    }
    assert(std::abs(f(x)) < eps);
    return x;
  }
  class SpeedChange {
   public:
    explicit SpeedChange(State const& before)
        : rho_before_(before.rho), p_before_(before.p),
          a_before_(before.rho == 0 ? 0 : Gas::GetSpeedOfSound(before)),
          p_const_(before.p * Gas::GammaMinusOneOverTwo()) {
      assert(before.rho >= 0 && before.p >= 0);
    }
    double operator()(double p_after) const {
      double value;
      assert(p_after >= 0);
      if (p_after >= p_before_) {  // shock
        value = (p_after - p_before_) / std::sqrt(rho_before_ * P(p_after));
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
        value = (1 + value / p) / std::sqrt(rho_before_ * p);
      } else {  // expansion
        constexpr double exp = -Gas::GammaPlusOneOverTwo() / Gas::Gamma();
        value = std::pow(p_after / p_before_, exp) / (rho_before_ * a_before_);
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
    Shock(State const& before, State const& after) : u(before.u) {
      u += (after.p - before.p) / ((after.u - before.u) * before.rho);
    }
    double GetDensityAfterIt(State const& before, State const& after) const {
      return before.rho * (before.u - u) / (after.u - u);
    }
  };
  template <int kField>
  static bool TimeAxisAfterWave(Shock<kField> const& wave);
  template <>
  static bool TimeAxisAfterWave(Shock<1> const& wave) { return wave.u < 0; }
  template <>
  static bool TimeAxisAfterWave(Shock<3> const& wave) { return wave.u > 0; }
  template <int kField>
  static State GetStateNearShock(State const& before, State* after) {
    static_assert(kField == 1 || kField == 3);
    auto shock = Shock<kField>(before, *after);
    if (TimeAxisAfterWave(shock)) {  // i.e. (x=0, t) is AFTER the shock.
      after->rho = before.rho * (before.u - shock.u) / (after->u - shock.u);
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
          gri_1(before.p / (std::pow(before.rho, Gas::Gamma()))) {
      gri_2 = AddOrMinus<kField>(
        before.u, a_before * Gas::GammaMinusOneUnderTwo());
      a_after = AddOrMinus<kField>(
        a_before, (before.u - after.u) * Gas::GammaMinusOneOverTwo());
    }
  };
  template <int kField>
  static constexpr double AddOrMinus(double x, double y);
  template <>
  static constexpr double AddOrMinus<1>(double x, double y) { return x + y; }
  template <>
  static constexpr double AddOrMinus<3>(double x, double y) { return x - y; }
  template <int kField>
  static bool TimeAxisAfterWave(double u, double a);
  template <>
  static bool TimeAxisAfterWave<1>(double u, double a) { return u - a < 0; }
  template <>
  static bool TimeAxisAfterWave<3>(double u, double a) { return u + a > 0; }
  template <int kField>
  static bool TimeAxisBeforeWave(double u, double a);
  template <>
  static bool TimeAxisBeforeWave<1>(double u, double a) { return u - a > 0; }
  template <>
  static bool TimeAxisBeforeWave<3>(double u, double a) { return u + a < 0; }
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
    if (TimeAxisAfterWave<kField>(after->u, wave.a_after)) {
      after->rho = Gas::Gamma() * after->p / (wave.a_after * wave.a_after);
      return *after;
    } else if (TimeAxisBeforeWave<kField>(before.u, wave.a_before)) {
      return before;
    } else {  // Axis[t] is inside the Expansion.
      return GetStateInsideExpansion(wave.gri_1, wave.gri_2);
    }
  }
  static State GetStateNearVaccum(State const& left, State const& right) {
    auto left_a = Gas::GetSpeedOfSound(left);
    if (left.u > left_a) {  // Axis[t] <<< Wave[1].
      return left;
    } else if (right.u + left_a < 0) {  // Wave[3] <<< Axis[t].
      return right;
    } else {  // Wave[1] <<< Axis[t] <<< Wave[3].
      auto gri_1 = left.p / (std::pow(left.rho, Gas::Gamma()));
      auto gri_2 = left.u + left_a * Gas::GammaMinusOneUnderTwo();
      if (gri_2 >= 0) {  // Axis[t] is inside Wave[1].
        return GetStateInsideExpansion(gri_1, gri_2);
      } else {  // gri_2 < 0
        // Axis[t] is to the RIGHT of Wave[1].
        gri_1 = right.p / (std::pow(right.rho, Gas::Gamma()));
        auto a_right = Gas::GetSpeedOfSound(right);
        gri_2 = right.u - a_right * Gas::GammaMinusOneUnderTwo();
        if (gri_2 < 0) {  // Axis[t] is inside Wave[3].
          return GetStateInsideExpansion(gri_1, gri_2);
        } else {  // Axis[t] is inside the vaccumed region.
          return {0, 0, 0};
        }
      }
    }
  }
};

}  // namespace euler
}  // namespace riemann
}  // namespace mini

#endif  //  MINI_RIEMANN_EULER_EXACT_HPP_
