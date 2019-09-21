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
  // Get F on t-Axis
  Flux GetFluxOnTimeAxis(State const& left, State const& right) {
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
          return GetFluxNearShock<1>(left, &star);
        } else {  // star.p < left.p : Wave[1] is an expansion.
          auto gri_1 = left.p / (std::pow(left.rho, Gas::Gamma()));
          auto left_a = Gas::GetSpeedOfSound(left);
          auto gri_2 = left.u + left_a * Gas::GammaMinusOneUnderTwo();
          auto star_a = left_a;
          star_a += (left.u - star.u) * Gas::GammaMinusOneOverTwo();
          if (star.u < star_a) {
            // Wave[1] <<< Axis[t] <<< Wave[2].
            star.rho = Gas::Gamma() * star.p / (star_a * star_a);
            return GetFlux(star);
          } else if (left.u > left_a) {
            // Axis[t] <<< Wave[1].
            return GetFlux(left);
          } else {  // Axis[t] is inside Wave[1].
            constexpr auto r = Gas::GammaMinusOne() / Gas::GammaPlusOne();
            auto a = r * gri_2;
            auto rho = std::pow(a * a / gri_1 * Gas::OneOverGamma(),
                                Gas::OneOverGammaMinusOne());
            return GetFlux({rho, a, a * a * rho * Gas::OneOverGamma()});
          }
        }
      } else {  // star.u < 0, so Axis[t] is BETWEEN Wave[2] and Wave[3].
        if (star.p >= right.p) {  // Wave[3] is a shock.
          return GetFluxNearShock<3>(right, &star);
        } else {  // star.p < right.p
          // Wave[3] is an expansion.
          auto gri_1 = right.p / (std::pow(right.rho, Gas::Gamma()));
          auto a_right = Gas::GetSpeedOfSound(right);
          auto gri_2 = right.u - a_right * Gas::GammaMinusOneUnderTwo();
          auto star_a = a_right;
          star_a += (star.u - right.u) * Gas::GammaMinusOneOverTwo();
          if (star.u + star_a > 0) {
            // Axis[t] is BETWEEN Wave[3] and Wave[2].
            star.rho = Gas::Gamma() * star.p / (star_a * star_a);
            return GetFlux({star.rho, star.u, star.p});
          } else if (right.u + a_right < 0) {
            // Axis[t] is to the RIGHT of Wave[3].
            return GetFlux(right);
          } else {  // Axis[t] is inside Wave[3].
            constexpr auto r = -Gas::GammaMinusOne() / Gas::GammaPlusOne();
            auto a = r * gri_2;
            auto rho = std::pow(a * a / gri_1 * Gas::OneOverGamma(),
                                Gas::OneOverGammaMinusOne());
            return GetFlux({rho, -a, a * a * rho * Gas::OneOverGamma()});
          }
        }
      }
    } else {  // The region BETWEEN Wave[1] and Wave[3] is vaccumed.
      auto left_a = std::sqrt(Gas::Gamma() * left.p / left.rho);
      if (left.u > left_a) {  // Axis[t] <<< Wave[1].
        return GetFlux(left);
      } else if (right.u + left_a < 0) {  // Wave[3] <<< Axis[t].
        return GetFlux(right);
      } else {  // Wave[1] <<< Axis[t] <<< Wave[3].
        auto gri_1 = left.p / (std::pow(left.rho, Gas::Gamma()));
        auto left_a = Gas::GetSpeedOfSound(left);
        auto gri_2 = left.u + left_a * Gas::GammaMinusOneUnderTwo();
        if (gri_2 >= 0) {  // Axis[t] is inside Wave[1].
          constexpr auto r = Gas::GammaMinusOne() / Gas::GammaPlusOne();
          auto a = r * gri_2;
          auto rho = std::pow(a * a / gri_1 * Gas::OneOverGamma(),
                              Gas::OneOverGammaMinusOne());
          return GetFlux({rho, a, a * a * rho * Gas::OneOverGamma()});
        } else {  // gri_2 < 0
          // Axis[t] is to the RIGHT of Wave[1].
          gri_1 = right.p / (std::pow(right.rho, Gas::Gamma()));
          auto a_right = Gas::GetSpeedOfSound(right);
          gri_2 = right.u - a_right * Gas::GammaMinusOneUnderTwo();
          if (gri_2 < 0) {  // Axis[t] is inside Wave[3].
            constexpr auto r = -Gas::GammaMinusOne() / Gas::GammaPlusOne();
            auto a = r * gri_2;
            auto rho = std::pow(a * a / gri_1 * Gas::OneOverGamma(),
                                Gas::OneOverGammaMinusOne());
            return GetFlux({rho, -a, a * a * rho * Gas::OneOverGamma()});
          } else {
            // Axis[t] is inside the vaccumed region.
            return {0, 0, 0};
          }
        }
      }
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
  static Flux GetFluxNearShock(State const& before, State* after) {
    static_assert(kField == 1 || kField == 3);
    auto shock = Shock<kField>(before, *after);
    if (TimeAxisAfterWave(shock)) {  // i.e. (x=0, t) is AFTER the shock.
      after->rho = before.rho * (before.u - shock.u) / (after->u - shock.u);
      return GetFlux(*after);
    } else {
      return GetFlux(before);
    }
  }
};

}  // namespace euler
}  // namespace riemann
}  // namespace mini

#endif  //  MINI_RIEMANN_EULER_EXACT_HPP_
