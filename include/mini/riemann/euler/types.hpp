//  Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_RIEMANN_EULER_TYPES_HPP_
#define MINI_RIEMANN_EULER_TYPES_HPP_

#include <cassert>
#include <cmath>
#include <cstring>

#include "mini/algebra/eigen.hpp"

namespace mini {
namespace riemann {
namespace euler {

template <int kDim, class ScalarType = double>
class Tuple : public algebra::Vector<ScalarType, kDim+2> {
  using Base = algebra::Vector<ScalarType, kDim+2>;

 public:
  // Types:
  using Scalar = ScalarType;
  using Vector = algebra::Vector<Scalar, kDim>;
  // Constructors:
  using Base::Base;
  // Accessors and Mutators:
  const Scalar& mass() const {
    return (*this)[0];
  }
  const Vector& momentum() const {
    return *reinterpret_cast<const Vector*>(&momentumX());
  }
  const Scalar& momentumX() const {
    return (*this)[1];
  }
  const Scalar& momentumY() const {
    assert(kDim >= 2);
    return (*this)[2];
  }
  const Scalar& momentumZ() const {
    assert(kDim == 3);
    return (*this)[3];
  }
  const Scalar& energy() const {
    return (*this)[kDim+1];
  }
  Scalar& mass() {
    return (*this)[0];
  }
  Vector& momentum() {
    return *reinterpret_cast<Vector*>(&momentumX());
  }
  Scalar& momentumX() {
    return (*this)[1];
  }
  Scalar& momentumY() {
    assert(kDim >= 2);
    return (*this)[2];
  }
  Scalar& momentumZ() {
    assert(kDim == 3);
    return (*this)[3];
  }
  Scalar& energy() {
    return (*this)[kDim+1];
  }
};

template <int kDim, class Scalar>
class Converter;

template <class Scalar>
class Converter<1, Scalar> {
 public:
  static void VelocityToMomentum(const Scalar &rho, Tuple<1, Scalar> *tuple) {
    tuple->momentumX() *= rho;
  }
  static void MomentumToVelocity(const Scalar &rho, Tuple<1, Scalar> *tuple) {
    tuple->momentumX() /= rho;
  }
};

template <class Scalar>
class Converter<2, Scalar> {
 public:
  static void VelocityToMomentum(const Scalar &rho, Tuple<2, Scalar> *tuple) {
    tuple->momentumX() *= rho;
    tuple->momentumY() *= rho;
  }
  static void MomentumToVelocity(const Scalar &rho, Tuple<2, Scalar> *tuple) {
    tuple->momentumX() /= rho;
    tuple->momentumY() /= rho;
  }
};

template <class Scalar>
class Converter<3, Scalar> {
 public:
  static void VelocityToMomentum(const Scalar &rho, Tuple<3, Scalar> *tuple) {
    tuple->momentumX() *= rho;
    tuple->momentumY() *= rho;
    tuple->momentumZ() *= rho;
  }
  static void MomentumToVelocity(const Scalar &rho, Tuple<3, Scalar> *tuple) {
    tuple->momentumX() /= rho;
    tuple->momentumY() /= rho;
    tuple->momentumZ() /= rho;
  }
};


template <int kDim, class ScalarType = double>
class FluxTuple : public Tuple<kDim, ScalarType> {
  using Base = Tuple<kDim, ScalarType>;
  using Mat5x1 = algebra::Matrix<ScalarType, 5, 1>;

 public:
  // Types:
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using FluxMatrix = algebra::Matrix<Scalar, 5, kDim>;
  // Constructors:
  using Base::Base;
};

template <int kDim, class ScalarType = double>
class PrimitiveTuple : public Tuple<kDim, ScalarType> {
  using Base = Tuple<kDim, ScalarType>;

 public:
  // Types:
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using Density = Scalar;
  using Pressure = Scalar;
  using Speed = Scalar;
  // Constructors:
  using Base::Base;
  // Accessors and Mutators:
  const Scalar& rho() const {
    return this->mass();
  }
  const Scalar& u() const {
    return this->momentumX();
  }
  const Scalar& v() const {
    return this->momentumY();
  }
  const Scalar& w() const {
    return this->momentumZ();
  }
  const Scalar& p() const {
    return this->energy();
  }
  Scalar& rho() {
    return this->mass();
  }
  Scalar& u() {
    return this->momentumX();
  }
  Scalar& v() {
    return this->momentumY();
  }
  Scalar& w() {
    return this->momentumZ();
  }
  Scalar& p() {
    return this->energy();
  }
  Scalar GetDynamicPressure() const {
    auto e_k = u()*u() + (kDim < 2 ? 0 : v()*v() + (kDim < 3 ? 0 : w()*w()));
    e_k *= 0.5;
    return rho() * e_k;
  }
};

template <int kDim, class ScalarType = double>
struct ConservativeTuple : public Tuple<kDim, ScalarType> {
  using Base = Tuple<kDim, ScalarType>;

  // Types:
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using Density = Scalar;
  using Pressure = Scalar;
  using Speed = Scalar;
  // Constructors:
  using Base::Base;
};

template <class ScalarType, int kInteger = 1, int kDecimal = 4>
class IdealGas {
 public:
  using Scalar = ScalarType;

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
  static double GetSpeedOfSound(PrimitiveTuple<kDim, Scalar> const& state) {
    return state.rho() == 0 ? 0 : std::sqrt(Gamma() * state.p() / state.rho());
  }
  template <int kDim>
  static PrimitiveTuple<kDim, Scalar> ConservativeToPrimitive(
      ConservativeTuple<kDim, Scalar> const &conservative) {
    auto primitive = PrimitiveTuple<kDim, Scalar>(conservative);
    if (primitive.rho() > 0) {
      Converter<kDim, Scalar>::MomentumToVelocity(primitive.rho(), &primitive);
      primitive.energy() -= primitive.GetDynamicPressure();
      primitive.energy() *= primitive.energy() < 0 ? 0 : GammaMinusOne();
    } else {
      primitive.setZero();
    }
    return primitive;
  }
  template <int kDim>
  static ConservativeTuple<kDim, Scalar> PrimitiveToConservative(
      PrimitiveTuple<kDim, Scalar> const &primitive) {
    auto conservative = ConservativeTuple<kDim, Scalar>(primitive);
    Converter<kDim, Scalar>::VelocityToMomentum(primitive.rho(), &conservative);
    conservative.energy() *= OneOverGammaMinusOne();  // p / (gamma - 1)
    conservative.energy() += primitive.GetDynamicPressure();
    return conservative;
  }
  static auto GetFluxMatrix(ConservativeTuple<3, Scalar> const& cv) {
    using FluxMatrix = typename FluxTuple<3, Scalar>::FluxMatrix;
    FluxMatrix mat;
    auto pv = ConservativeToPrimitive(cv);
    auto rho = pv.rho(), u = pv.u(), v = pv.v(), w = pv.w(), p = pv.p();
    auto rho_u = cv.momentumX(), rho_v = cv.momentumY(), rho_w = cv.momentumZ();
    auto rho_h0 = cv.energy() + p;
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
