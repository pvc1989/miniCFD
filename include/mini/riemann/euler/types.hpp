//  Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_RIEMANN_EULER_TYPES_HPP_
#define MINI_RIEMANN_EULER_TYPES_HPP_

#include <cmath>
#include <initializer_list>

#include "mini/algebra/column.hpp"

namespace mini {
namespace riemann {
namespace euler {

template <int kDim>
class Tuple {
 public:
  // Types:
  using Scalar = double;
  using Vector = algebra::Column<Scalar, kDim>;
  // Data:
  Scalar mass{0};
  Vector momentum{0};
  Scalar energy{0};
  // Constructors:
  Tuple() = default;
  Tuple(Scalar const& rho,
        Scalar const& u,
        Scalar const& p)
      : mass{rho}, energy{p}, momentum{u} {}
  Tuple(Scalar const& rho,
        Scalar const& u, Scalar const& v,
        Scalar const& p)
      : mass{rho}, energy{p}, momentum{u, v} {}
  // Arithmetic Operators:
  Tuple& operator+=(Tuple const& that) {
    this->mass += that.mass;
    this->energy += that.energy;
    this->momentum += that.momentum;
    return *this;
  }
  Tuple& operator-=(Tuple const& that) {
    this->mass -= that.mass;
    this->energy -= that.energy;
    this->momentum -= that.momentum;
    return *this;
  }
  Tuple& operator*=(Scalar const& s) {
    this->mass *= s;
    this->energy *= s;
    this->momentum *= s;
    return *this;
  }
  Tuple& operator/=(Scalar const& s) {
    this->mass /= s;
    this->energy /= s;
    this->momentum /= s;
    return *this;
  }
  // Other Operations:
  bool operator==(Tuple const& that) const {
    return (this->mass == that.mass) && (this->energy == that.energy) &&
           (this->momentum == that.momentum);
  }
};
template <int kDim>
class Flux : public Tuple<kDim> {
  // Types:
  using Base = Tuple<kDim>;
  // Constructors:
  using Base::Base;
};
template <int kDim>
class Primitive : public Tuple<kDim> {
 public:
  // Types:
  using Base = Tuple<kDim>;
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using Density = Scalar;
  using Pressure = Scalar;
  using Speed = Scalar;
  // Constructors:
  using Base::Base;
  // Accessors and Mutators:
  Density const& rho() const { return this->mass; }
  Pressure const& p() const { return this->energy; }
  Speed const& u() const { return this->momentum[0]; }
  Speed const& v() const { return this->momentum[1]; }
  Density& rho() { return this->mass; }
  Pressure& p() { return this->energy; }
  Speed& u() { return this->momentum[0]; }
  Speed& v() { return this->momentum[1]; }
};
template <int kDim>
struct Conservative : Tuple<kDim>{
  // Types:
  using Base = Tuple<kDim>;
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using Density = Scalar;
  using Pressure = Scalar;
  using Speed = Scalar;
  // Constructors:
  using Base::Base;
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
  // Converters:
  template <int kDim>
  static double GetSpeedOfSound(Primitive<kDim> const& state) {
    return state.rho() == 0 ? 0 : std::sqrt(Gamma() * state.p() / state.rho());
  }
  template <int kDim>
  static Primitive<kDim>& ConservativeToPrimitive(Conservative<kDim>* state) {
    auto& rho = state->mass;
    if (rho) {
      // momentum = rho * u
      state->momentum /= rho;
      auto& u = state->momentum;
      // energy = p/(gamma - 1) + 0.5*rho*|u|^2
      state->energy -= 0.5 * rho * u.Dot(u);
      state->energy *= GammaMinusOne();
    }
    return reinterpret_cast<Primitive<kDim>&>(*state);
  }
  template <int kDim>
  static Conservative<kDim>& PrimitiveToConservative(Primitive<kDim>* state) {
    auto& rho = state->mass;
    auto& u = state->momentum;
    // energy = p/(gamma - 1) + 0.5*rho*|u|^2
    state->energy *= OneOverGammaMinusOne();  // p / (gamma - 1)
    state->energy += 0.5 * rho * u.Dot(u);  // + 0.5 * rho * |u|^2
    // momentum = rho * u
    state->momentum *= rho;
    return reinterpret_cast<Conservative<kDim>&>(*state);
  }
};

}  // namespace euler
}  // namespace riemann
}  // namespace mini

#endif  //  MINI_RIEMANN_EULER_TYPES_HPP_
