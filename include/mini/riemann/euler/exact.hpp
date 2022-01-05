//  Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_RIEMANN_EULER_EXACT_HPP_
#define MINI_RIEMANN_EULER_EXACT_HPP_

#include <cassert>
#include <cmath>

#include "mini/riemann/euler/types.hpp"

namespace mini {
namespace riemann {
namespace euler {

template <class Gas, int D>
class Implementor {
 public:
  constexpr static int kDim = D;
  // Types:
  using Scalar = typename Gas::Scalar;
  using Flux = FluxTuple<Scalar, kDim>;
  using Conservative = ConservativeTuple<Scalar, kDim>;
  using Primitive = PrimitiveTuple<Scalar, kDim>;
  using Vector = typename Primitive::Vector;
  using Speed = Scalar;
  // Data:
  Speed star_u{0.0};
  // Get U on t-Axis
  Primitive PrimitiveOnTimeAxis(const Primitive& left, const Primitive& right) {
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
      auto star = Primitive();
      star.p() = FindRoot(f, f_prime, left.p());
      star.u() = 0.5 * (right.u() + u_change_right(star.p())
                        +left.u() - u_change__left(star.p()));
      star_u = star.u();
      if (0 < star.u()) {  // Axis[t] <<< Wave[2]
        if (star.p() >= left.p()) {  // Wave[1] is a shock.
          return PrimitiveNearShock<1>(left, &star);
        } else {  // star.p() < left.p() : Wave[1] is an expansion.
          return PrimitiveNearExpansion<1>(left, &star);
        }
      } else {  // star.u() < 0 : Wave[2] <<< Axis[t] <<< Wave[3].
        if (star.p() >= right.p()) {  // Wave[3] is a shock.
          return PrimitiveNearShock<3>(right, &star);
        } else {  // star.p() < right.p() : Wave[3] is an expansion.
          return PrimitiveNearExpansion<3>(right, &star);
        }
      }
    } else {  // The region BETWEEN Wave[1] and Wave[3] is vacuumed.
      return PrimitiveNearVacuum(left, right);
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
    explicit SpeedChange(const Primitive& before)
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
    Shock(const Primitive& before, const Primitive& after) : u(before.u()) {
      auto divisor = (after.u() - before.u()) * before.rho();
      u += (divisor ? (after.p() - before.p()) / divisor
                    : before.u());
    }
    double GetDensityAfterIt(const Primitive& before, const Primitive& after) const {
      auto divisor = after.u() - u;
      return divisor ? before.rho() * (before.u() - u) / divisor
                     : before.rho();
    }
  };
  static bool TimeAxisAfterShock(Shock<1> const& wave) { return wave.u < 0; }
  static bool TimeAxisAfterShock(Shock<3> const& wave) { return wave.u > 0; }
  template <int kField>
  static Primitive PrimitiveNearShock(const Primitive& before, Primitive* after) {
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
    static constexpr double AddOrMinus(double x, double y) {
      static_assert(kField == 1 || kField == 3);
      return kField == 1 ? (x + y) : (x - y);
    }

   public:
    double a_before, a_after;  // speed of sound before/after the wave
    double gri_1, gri_2;  // Generalized Riemann Invariants
    Expansion(const Primitive& before, const Primitive& after)
        : a_before(Gas::GetSpeedOfSound(before)),
          gri_1(before.p() / (std::pow(before.rho(), Gas::Gamma()))) {
      gri_2 = AddOrMinus(before.u(),
          Gas::GammaMinusOneUnderTwo() * a_before);
      a_after = AddOrMinus(a_before,
          Gas::GammaMinusOneOverTwo() * (before.u() - after.u()));
    }
  };
  static Primitive PrimitiveInsideExpansion(double gri_1, double gri_2) {
    constexpr auto r = Gas::GammaMinusOne() / Gas::GammaPlusOne();
    auto a = r * gri_2;
    auto a_square = a * a;
    auto rho = std::pow(a_square / gri_1 * Gas::OneOverGamma(),
                        Gas::OneOverGammaMinusOne());
    auto state = Primitive();
    state.rho() = rho;
    state.u() = a;
    state.p() = a_square * rho * Gas::OneOverGamma();
    return state;
  }
  template <int kField>
  constexpr static bool TimeAxisAfterExpansion(double u, double a) {
    static_assert(kField == 1 || kField == 3);
    return kField == 1 ? (u - a < 0) : (u + a > 0);
  }
  template <int kField>
  static bool TimeAxisBeforeExpansion(double u, double a) {
    static_assert(kField == 1 || kField == 3);
    return kField == 1 ? (u - a > 0) : (u + a < 0);
  }
  template <int kField>
  static Primitive PrimitiveNearExpansion(const Primitive& before, Primitive* after) {
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
      return PrimitiveInsideExpansion(wave.gri_1, wave.gri_2);
    }
  }
  static Primitive PrimitiveNearVacuum(const Primitive& left, const Primitive& right) {
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
        return PrimitiveInsideExpansion(gri_1, gri_2);
      } else {  // gri_2 < 0
        // Axis[t] is to the RIGHT of Wave[1].
        gri_1 = right.p() / (std::pow(right.rho(), Gas::Gamma()));
        gri_2 = right.u() - right_a * Gas::GammaMinusOneUnderTwo();
        if (gri_2 < 0) {  // Axis[t] is inside Wave[3].
          return PrimitiveInsideExpansion(gri_1, gri_2);
        } else {  // Axis[t] is inside the vacuumed region.
          auto state = Primitive();
          state.setZero();
          return state;
        }
      }
    }
  }
};

template <class GasType, int kDim>
class Exact;

template <class GasType>
class Exact<GasType, 1> : public Implementor<GasType, 1> {
  using Base = Implementor<GasType, 1>;

 public:
  constexpr static int kDim = Base::kDim;
  // Types:
  using Gas = GasType;
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using Flux = typename Base::Flux;
  using Conservative = typename Base::Conservative;
  using Primitive = typename Base::Primitive;
  // Get F from U
  static Flux GetFlux(const Primitive& state) {
    auto rho_u = state.rho() * state.u();
    auto rho_u_u = rho_u * state.u();
    return {rho_u, rho_u_u + state.p(),
            state.u() * (state.p() * Gas::GammaOverGammaMinusOne()
                         + 0.5 * rho_u_u)};
  }
  // Get F on t-Axis
  Flux GetFluxOnTimeAxis(const Primitive& left, const Primitive& right) {
    return GetFlux(PrimitiveOnTimeAxis(left, right));
  }
  // Get U on t-Axis
  Primitive PrimitiveOnTimeAxis(const Primitive& left, const Primitive& right) {
    return Base::PrimitiveOnTimeAxis(left, right);
  }
};
template <class GasType>
class Exact<GasType, 2> : public Implementor<GasType, 2> {
  using Base = Implementor<GasType, 2>;

 public:
  constexpr static int kDim = Base::kDim;
  // Types:
  using Gas = GasType;
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using Flux = typename Base::Flux;
  using Conservative = typename Base::Conservative;
  using Primitive = typename Base::Primitive;
  // Get F from U
  static Flux GetFlux(const Primitive& state) {
    auto rho_u = state.rho() * state.u();
    auto rho_v = state.rho() * state.v();
    auto rho_u_u = rho_u * state.u();
    return {rho_u, rho_u_u + state.p(), rho_v * state.u(),
            state.u() * (state.p() * Gas::GammaOverGammaMinusOne()
                         + 0.5 * (rho_u_u + rho_v * state.v()))};
  }
  // Get F on t-Axis
  Flux GetFluxOnTimeAxis(const Primitive& left, const Primitive& right) {
    auto state = PrimitiveOnTimeAxis(left, right);
    return GetFlux(state);
  }
  // Get U on t-Axis
  Primitive PrimitiveOnTimeAxis(const Primitive& left, const Primitive& right) {
    auto state = Base::PrimitiveOnTimeAxis(left, right);
    state.v() = this->star_u > 0 ? left.v() : right.v();
    return state;
  }
};
template <class GasType>
class Exact<GasType, 3> : public Implementor<GasType, 3> {
  using Base = Implementor<GasType, 3>;

 public:
  constexpr static int kDim = Base::kDim;
  // Types:
  using Gas = GasType;
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using Flux = typename Base::Flux;
  using Conservative = typename Base::Conservative;
  using Primitive = typename Base::Primitive;
  // Get F from U
  static Flux GetFlux(const Primitive& state) {
    auto rho_u = state.rho() * state.u();
    auto rho_v = state.rho() * state.v();
    auto rho_w = state.rho() * state.w();
    auto rho_u_u = rho_u * state.u();
    return {rho_u, rho_u_u + state.p(), rho_v * state.u(), rho_w * state.u(),
            state.u() * (state.p() * Gas::GammaOverGammaMinusOne()
            + 0.5 * (rho_u_u + rho_v * state.v() + rho_w * state.w()))};
  }
  // Get F on t-Axis
  Flux GetFluxOnTimeAxis(const Primitive& left, const Primitive& right) {
    return GetFlux(PrimitiveOnTimeAxis(left, right));
  }
  // Get U on t-Axis
  Primitive PrimitiveOnTimeAxis(const Primitive& left, const Primitive& right) {
    auto state = Base::PrimitiveOnTimeAxis(left, right);
    state.v() = this->star_u > 0 ? left.v() : right.v();
    state.w() = this->star_u > 0 ? left.w() : right.w();
    return state;
  }
};

}  // namespace euler
}  // namespace riemann
}  // namespace mini

#endif  //  MINI_RIEMANN_EULER_EXACT_HPP_
