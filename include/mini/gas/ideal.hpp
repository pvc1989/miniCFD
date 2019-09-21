//  Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_GAS_IDEAL_HPP_
#define MINI_GAS_IDEAL_HPP_

#include <cmath>

namespace mini {
namespace gas {

template <int kDim>
class State;
template <>
class State<1> {
 public:
  // Types:
  using Density = double;
  using Speed = double;
  using Pressure = double;
  // Data:
  Density rho;
  Speed u;
  Pressure p;
  // Constructors:
  State(Density rho, Speed u, Pressure p) : rho(rho), u(u), p(p) {}
};
template <>
class State<2> : public State<1> {
 public:
  // Data:
  Speed v;
  // Constructors:
  State(Density rho, Speed u, Pressure p, Speed v = 0.0)
      : State<1>(rho, u, p), v(v) {}
};

template <int kInteger = 1, int kDecimal = 4>
class Ideal {
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
    return state.rho == 0 ? 0 : std::sqrt(Gamma() * state.p / state.rho);
  }
};

}  // namespace gas
}  // namespace mini

#endif  //  MINI_GAS_IDEAL_HPP_
