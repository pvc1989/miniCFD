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
  // Get F on t-Axis
  Flux GetFluxOnTimeAxis(State const& left, State const& right) {
    // Determine p_2
    auto u_change_given = right.u - left.u;
    auto u_change_left  = SpeedChange(left);
    auto u_change_right = SpeedChange(right);
    auto f = [&](double p) {
      return u_change_left(p) + u_change_right(p) + u_change_given;
    };
    auto f_prime = [&](double p) {
      return u_change_left.Prime(p) + u_change_right.Prime(p);
    };
    if (f(0) < 0) {  // Ordinary case: Wave[2] is a contact.
      auto p_2 = left.p;
      while (f(p_2) > 0) { p_2 *= 0.5; }
      p_2 = FindRoot(f, f_prime, p_2);
      auto u_2 = left.u + right.u + u_change_right(p_2) - u_change_left(p_2);
      u_2 *= 0.5;
      if (u_2 > 0) {  // Axis[t] is to the left of Wave[2].
        if (p_2 >= left.p) {  // Wave[1] is a shock.
          auto u_shock = left.u + (p_2 - left.p) / ((u_2 - left.u) * left.rho);
          if (u_shock >= 0) {  // Axis[t] is to the left of Wave[1].
            return GetFlux(left);
          } else {  // Axis[t] is between Wave[1] and Wave[2].
            auto rho_2 = left.rho * (left.u - u_shock) / (u_2 - u_shock);
            return GetFlux({rho_2, u_2, p_2});
          }
        } else {  // p_2 < left.p : Wave[1] is an expansion.
          auto gri_1 = left.p / (std::pow(left.rho, Gas::Gamma()));
          auto a_left = Gas::GetSpeedOfSound(left);
          auto gri_2 = left.u + a_left * Gas::GammaMinusOneUnderTwo();
          auto a_2 = a_left + (left.u - u_2) * Gas::GammaMinusOneOverTwo();
          if (u_2 < a_2) {
            // Axis[t] is between Wave[1] and Wave[2].
            auto rho_2 = Gas::Gamma() * p_2 / (a_2 * a_2);
            return GetFlux({rho_2, u_2, p_2});
          } else if (left.u > a_left) {
            // Axis[t] is to the left of Wave[1].
            return GetFlux(left);
          } else {  // Axis[t] is inside Wave[1].
            constexpr auto r = Gas::GammaMinusOne() / Gas::GammaPlusOne();
            auto a = r * gri_2;
            auto rho = std::pow(a * a / gri_1 * Gas::OneOverGamma(),
                                Gas::OneOverGammaMinusOne());
            return GetFlux({rho, a, a * a * rho * Gas::OneOverGamma()});
          }
        }
      } else {  // u_2 < 0, so Axis[t] is to the right of Wave[2].
        if (p_2 >= right.p) {  // Wave[3] is a shock.
          auto u_shock = right.u;
          u_shock += (p_2 - right.p) / ((u_2 - right.u) * right.rho);
          if (u_shock <= 0) {
            // Axis[t] is to the right of Wave[3].
            return GetFlux(right);
          } else {
            // Axis[t] is between Wave[2] and Wave[3].
            auto rho_2 = right.rho * (right.u - u_shock) / (u_2 - u_shock);
            return GetFlux({rho_2, u_2, p_2});
          }
        } else {  // p_2 < right.p
          // Wave[3] is an expansion.
          auto gri_1 = right.p / (std::pow(right.rho, Gas::Gamma()));
          auto a_right = Gas::GetSpeedOfSound(right);
          auto gri_2 = right.u - a_right * Gas::GammaMinusOneUnderTwo();
          auto a_2 = a_right + (u_2 - right.u) * Gas::GammaMinusOneOverTwo();
          if (u_2 + a_2 > 0) {
            // Axis[t] is between Wave[3] and Wave[2].
            auto rho_2 = Gas::Gamma() * p_2 / (a_2 * a_2);
            return GetFlux({rho_2, u_2, p_2});
          } else if (right.u + a_right < 0) {
            // Axis[t] is to the right of Wave[3].
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
    } else {  // The region between Wave[1] and Wave[3] is vaccumed.
      double rho_2{0}, u_2{0}, p_2{0};
      auto a_left = std::sqrt(Gas::Gamma() * left.p / left.rho);
      if (left.u > a_left) {  // Axis[t] is to the left of Wave[1].
        return GetFlux(left);
      } else if (right.u + a_left < 0) {  // Axis[t] is to the right of Wave[3].
        return GetFlux(right);
      } else {  // Axis[t] is to between Wave[1] and Wave[3].
        auto gri_1 = left.p / (std::pow(left.rho, Gas::Gamma()));
        auto a_left = Gas::GetSpeedOfSound(left);
        auto gri_2 = left.u + a_left * Gas::GammaMinusOneUnderTwo();
        if (gri_2 >= 0) {  // Axis[t] is inside Wave[1].
          constexpr auto r = Gas::GammaMinusOne() / Gas::GammaPlusOne();
          auto a = r * gri_2;
          auto rho = std::pow(a * a / gri_1 * Gas::OneOverGamma(),
                              Gas::OneOverGammaMinusOne());
          return GetFlux({rho, a, a * a * rho * Gas::OneOverGamma()});
        } else {  // gri_2 < 0
          // Axis[t] is to the right of Wave[1].
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
  // Get F from U
  Flux GetFlux(State const& state) {
    auto rho_u = state.rho * state.u;
    auto rho_u_u = rho_u * state.u;
    return {rho_u, rho_u_u + state.p,
            state.u * (state.p * Gas::GammaOverGammaMinusOne()
                       + 0.5 * rho_u_u)};
  }

 private:
  template <class F, class Fprime>
  static double FindRoot(F&& f, Fprime&& f_prime, double x, double eps = 1e-8) {
    assert(f(x) < 0);
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
};

}  // namespace euler
}  // namespace riemann
}  // namespace mini

#endif  //  MINI_RIEMANN_EULER_EXACT_HPP_
