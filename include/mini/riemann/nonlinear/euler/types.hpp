//  Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_RIEMANN_NONLINEAR_EULER_TYPES_HPP_
#define MINI_RIEMANN_NONLINEAR_EULER_TYPES_HPP_

#include <cmath>

namespace mini {
namespace riemann {
namespace nonlinear {
namespace euler {

template <int kDim>
class State;
template <>
class State<1> {
 public:
  // Types:
  using Scalar = double;
  using Density = double;
  using Speed = double;
  using Pressure = double;
  // Constructors:
  State(Density rho, Speed u, Pressure p) : rho_(rho), u_(u), p_(p) {}
  // Accessors and Mutators:
  Scalar const& rho() const { return rho_; }
  Scalar const& u() const { return u_; }
  Scalar const& p() const { return p_; }
  Scalar& rho() { return rho_; }
  Scalar& u() { return u_; }
  Scalar& p() { return p_; }
 protected:
  // Data:
  Density rho_;
  Speed u_;
  Pressure p_;
};
template <>
class State<2> : public State<1> {
 public:
  // Constructors:
  State(Density rho, Speed u, Speed v, Pressure p)
      : State<1>(rho, u, p), v_(v) {}
  State(Density rho, Speed u, Pressure p)
      : State(rho, u, 0.0, p) {}
  // Accessors and Mutators:
  Scalar const& v() const { return v_; }
  Scalar& v() { return v_; }
 protected:
  // Data:
  Speed v_;
};

template <int kInteger = 1, int kDecimal = 4>
class IdealGas {
 private:
  static_assert(kInteger >= 1 && kDecimal >= 0);
  static constexpr double Shift(double x) {
    return x < 1.0 ? x : Shift(x / 10.0);
  }
  static constexpr double gamma_ = kInteger + Shift(kDecimal);
 public:
  // Constants:
  static constexpr double Gamma() { return gamma_; }
  static constexpr double OneOverGamma() {
    return 1 / Gamma();
  }
  static constexpr double GammaPlusOne() {
    return Gamma() + 1;
  }
  static constexpr double GammaPlusOneOverTwo() {
    return GammaPlusOne() / 2;
  }
  static constexpr double GammaPlusOneOverFour() {
    return GammaPlusOne() / 4;
  }
  static constexpr double GammaMinusOne() {
    return Gamma() - 1;
  }
  static constexpr double OneOverGammaMinusOne() {
    return 1 / GammaMinusOne();
  }
  static constexpr double GammaOverGammaMinusOne() {
    return Gamma() / GammaMinusOne();
  }
  static constexpr double GammaMinusOneOverTwo() {
    return GammaMinusOne() / 2;
  }
  static constexpr double GammaMinusOneUnderTwo() {
    return 2 / GammaMinusOne();
  }
  // State equations:
  template <class State>
  static double GetSpeedOfSound(State const& state) {
    return state.rho() == 0 ? 0 : std::sqrt(Gamma() * state.p() / state.rho());
  }
};

}  // namespace euler
}  // namespace nonlinear
}  // namespace riemann
}  // namespace mini

#endif  //  MINI_RIEMANN_NONLINEAR_EULER_TYPES_HPP_
