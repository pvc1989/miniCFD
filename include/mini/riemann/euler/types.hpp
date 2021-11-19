//  Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_RIEMANN_EULER_TYPES_HPP_
#define MINI_RIEMANN_EULER_TYPES_HPP_

#include <cmath>
#include <initializer_list>

#include "mini/algebra/column.hpp"  // TODO(PVC): replace with EIGEN
#include "mini/algebra/eigen.hpp"

namespace mini {
namespace riemann {
namespace euler {

template <int kDim>
class Tuple {
 public:
  // Types:
  using Scalar = double;  // TODO(PVC): pass by template argument
  using Vector = algebra::Column<Scalar, kDim>;
  // Data:
  Scalar mass{0}, energy{0};
  Vector momentum{0};
  // Constructors:
  Tuple() = default;
  Tuple(Scalar const& rho,
        Scalar const& u,
        Scalar const& p)
      : mass{rho}, energy{p}, momentum{u} {
    static_assert(kDim >= 1);
  }
  Tuple(Scalar const& rho,
        Scalar const& u, Scalar const& v,
        Scalar const& p)
      : mass{rho}, energy{p}, momentum{u, v} {
    static_assert(kDim >= 2);
  }
  Tuple(Scalar const& rho,
        Scalar const& u, Scalar const& v, Scalar const& w,
        Scalar const& p)
      : mass{rho}, energy{p}, momentum{u, v, w} {
    static_assert(kDim == 3);
  }
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
class FluxTuple : public Tuple<kDim> {
 public:
  // Types:
  using Base = Tuple<kDim>;
  // Constructors:
  using Base::Base;

  operator algebra::Matrix<double, 5, 1>() const {
    static_assert(kDim == 3);
    return { this->mass, this->momentum[0], this->momentum[1], this->momentum[2], this->energy };
  }
};

template <int kDim>
class PrimitiveTuple : public Tuple<kDim> {
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
  explicit PrimitiveTuple(Base const& tuple) : Base(tuple) {}
  // Accessors and Mutators:
  Density const& rho() const { return this->mass; }
  Pressure const& p() const { return this->energy; }
  Speed const& u() const { return this->momentum[0]; }
  Speed const& v() const { return this->momentum[1]; }
  Speed const& w() const {
    static_assert(kDim == 3);
    return this->momentum[2];
  }
  Density& rho() { return this->mass; }
  Pressure& p() { return this->energy; }
  Speed& u() { return this->momentum[0]; }
  Speed& v() { return this->momentum[1]; }
  Speed& w() {
    static_assert(kDim == 3);
    return this->momentum[2];
  }
};

template <int kDim>
struct ConservativeTuple : public Tuple<kDim>{
  // Types:
  using Base = Tuple<kDim>;
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using Density = Scalar;
  using Pressure = Scalar;
  using Speed = Scalar;
  // Constructors:
  using Base::Base;
  explicit ConservativeTuple(Base const& tuple) : Base(tuple) {
  }

  ConservativeTuple(const algebra::Matrix<Scalar, kDim+2, 1>& v) {
    this->mass = v[0];
    this->energy = v[kDim+1];
    for (int i = 0; i < kDim; ++i) {
      this->momentum[i] = v[i+1];
    }
  }
};

template <int kInteger = 1, int kDecimal = 4>
class IdealGas {
 public:
  using Scalar = double;
  using FluxMatrix = algebra::Matrix<Scalar, 5, 3>;

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
  static double GetSpeedOfSound(PrimitiveTuple<kDim> const& state) {
    return state.rho() == 0 ? 0 : std::sqrt(Gamma() * state.p() / state.rho());
  }
  template <int kDim>
  static PrimitiveTuple<kDim>& ConservativeToPrimitive(Tuple<kDim>* state) {
    auto& rho = state->mass;
    if (rho > 0) {
      // momentum = rho * u
      state->momentum /= rho;
      auto& u = state->momentum;
      // energy = p/(gamma - 1) + 0.5*rho*|u|^2
      state->energy -= 0.5 * rho * u.Dot(u);
      state->energy *= GammaMinusOne();
      if (state->energy < 0) {
        assert(-0.0001 < state->energy);
        state->energy = 0;
      }
    } else {
      assert(rho == 0);
      state->momentum *= 0.0;
      state->energy = 0.0;
    }
    return reinterpret_cast<PrimitiveTuple<kDim>&>(*state);
  }
  template <int kDim>
  static PrimitiveTuple<kDim> ConservativeToPrimitive(
      ConservativeTuple<kDim> const& conservative) {
    auto primitive = PrimitiveTuple<kDim>{ conservative };
    ConservativeToPrimitive(&primitive);
    return primitive;
  }
  template <int kDim>
  static ConservativeTuple<kDim>& PrimitiveToConservative(Tuple<kDim>* state) {
    auto& rho = state->mass;
    auto& u = state->momentum;
    // energy = p/(gamma - 1) + 0.5*rho*|u|^2
    state->energy *= OneOverGammaMinusOne();  // p / (gamma - 1)
    state->energy += 0.5 * rho * u.Dot(u);  // + 0.5 * rho * |u|^2
    // momentum = rho * u
    state->momentum *= rho;
    return reinterpret_cast<ConservativeTuple<kDim>&>(*state);
  }
  template <int kDim>
  static ConservativeTuple<kDim> PrimitiveToConservative(
      PrimitiveTuple<kDim> const& primitive) {
    auto conservative = ConservativeTuple<kDim>{ primitive };
    PrimitiveToConservative(&conservative);
    return conservative;
  }
  static FluxMatrix GetFluxMatrix(ConservativeTuple<3> const& cv) {
    FluxMatrix mat;
    auto pv = ConservativeToPrimitive(cv);
    auto rho = pv.rho(), u = pv.u(), v = pv.v(), w = pv.w(), p = pv.p();
    auto rho_u = cv.momentum[0], rho_v = cv.momentum[1], rho_w = cv.momentum[2];
    auto rho_h0 = cv.energy + p;
    mat.col(0) << rho_u, rho_u * u + p, rho_v * u, rho_w * u, rho_h0 * u;
    mat.col(1) << rho_v, rho_u * v, rho_v * v + p, rho_w * v, rho_h0 * v;
    mat.col(2) << rho_w, rho_u * w, rho_v * w, rho_w * w + p, rho_h0 * w;
    return mat;
  }
};

}  // namespace euler
}  // namespace riemann
}  // namespace mini

#endif  //  MINI_RIEMANN_EULER_TYPES_HPP_
