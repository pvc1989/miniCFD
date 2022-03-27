//  Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_RIEMANN_EULER_TYPES_HPP_
#define MINI_RIEMANN_EULER_TYPES_HPP_

#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "mini/algebra/eigen.hpp"

namespace mini {
namespace riemann {
namespace euler {

template <class ScalarType, int kDim>
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

template <class Scalar, int kDim>
class Converter;

template <class Scalar>
class Converter<Scalar, 1> {
 public:
  static void VelocityToMomentum(const Scalar &rho, Tuple<Scalar, 1> *tuple) {
    tuple->momentumX() *= rho;
  }
  static void MomentumToVelocity(const Scalar &rho, Tuple<Scalar, 1> *tuple) {
    tuple->momentumX() /= rho;
  }
};

template <class Scalar>
class Converter<Scalar, 2> {
 public:
  static void VelocityToMomentum(const Scalar &rho, Tuple<Scalar, 2> *tuple) {
    tuple->momentumX() *= rho;
    tuple->momentumY() *= rho;
  }
  static void MomentumToVelocity(const Scalar &rho, Tuple<Scalar, 2> *tuple) {
    tuple->momentumX() /= rho;
    tuple->momentumY() /= rho;
  }
};

template <class Scalar>
class Converter<Scalar, 3> {
 public:
  static void VelocityToMomentum(const Scalar &rho, Tuple<Scalar, 3> *tuple) {
    tuple->momentumX() *= rho;
    tuple->momentumY() *= rho;
    tuple->momentumZ() *= rho;
  }
  static void MomentumToVelocity(const Scalar &rho, Tuple<Scalar, 3> *tuple) {
    tuple->momentumX() /= rho;
    tuple->momentumY() /= rho;
    tuple->momentumZ() /= rho;
  }
};


template <class ScalarType, int kDim>
class FluxTuple : public Tuple<ScalarType, kDim> {
  using Base = Tuple<ScalarType, kDim>;
  using Mat5x1 = algebra::Matrix<ScalarType, 5, 1>;

 public:
  // Types:
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using FluxMatrix = algebra::Matrix<Scalar, 5, kDim>;
  // Constructors:
  using Base::Base;
};

template <class ScalarType, int kDim>
class PrimitiveTuple : public Tuple<ScalarType, kDim> {
  using Base = Tuple<ScalarType, kDim>;

 public:
  // Types:
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
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

template <class ScalarType, int kDim>
struct ConservativeTuple : public Tuple<ScalarType, kDim> {
  using Base = Tuple<ScalarType, kDim>;

  // Types:
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
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

  static void AssertNonNegative(const Scalar &val, const char *name) {
    if (val < 0) {
      auto msg = name + std::string(" cannot be ") + std::to_string(val);
      throw std::runtime_error(msg);
    }
  }

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
  static double GetSpeedOfSound(PrimitiveTuple<Scalar, kDim> const& state) {
    AssertNonNegative(state.rho(), "density");
    return state.rho() == 0 ? 0 : std::sqrt(Gamma() * state.p() / state.rho());
  }
  template <int kDim>
  static PrimitiveTuple<Scalar, kDim> ConservativeToPrimitive(
      ConservativeTuple<Scalar, kDim> const &conservative) {
    auto primitive = PrimitiveTuple<Scalar, kDim>(conservative);
    AssertNonNegative(primitive.rho(), "density");
    if (primitive.rho()) {
      Converter<Scalar, kDim>::MomentumToVelocity(primitive.rho(), &primitive);
      primitive.energy() -= primitive.GetDynamicPressure();
      primitive.energy() *= GammaMinusOne();
      AssertNonNegative(primitive.energy(), "pressure");
    } else {
      primitive.setZero();
    }
    return primitive;
  }
  template <int kDim>
  static ConservativeTuple<Scalar, kDim> PrimitiveToConservative(
      PrimitiveTuple<Scalar, kDim> const &primitive) {
    auto conservative = ConservativeTuple<Scalar, kDim>(primitive);
    Converter<Scalar, kDim>::VelocityToMomentum(primitive.rho(), &conservative);
    conservative.energy() *= OneOverGammaMinusOne();  // p / (gamma - 1)
    conservative.energy() += primitive.GetDynamicPressure();
    return conservative;
  }
  template <int kDim>
  static FluxTuple<Scalar, kDim> PrimitiveToFlux(
      const PrimitiveTuple<Scalar, kDim> &primitive) {
    auto conservative = PrimitiveToConservative(primitive);
    conservative *= primitive.u();
    auto flux_x = FluxTuple<Scalar, kDim>(conservative);
    flux_x.momentumX() += primitive.p();
    flux_x.energy() += primitive.p() * primitive.u();
    return flux_x;
  }
  static auto GetFluxMatrix(ConservativeTuple<Scalar, 3> const& cv) {
    using FluxMatrix = typename FluxTuple<Scalar, 3>::FluxMatrix;
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
