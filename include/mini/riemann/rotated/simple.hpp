// Copyright 2019 PEI Weicheng and YANG Minghao
#ifndef MINI_RIEMANN_ROTATED_SIMPLE_HPP_
#define MINI_RIEMANN_ROTATED_SIMPLE_HPP_

#include <cstring>

#include "mini/algebra/eigen.hpp"
#include "mini/constant/index.hpp"

namespace mini {
namespace riemann {
namespace rotated {

using namespace mini::constant::index;

template <class UnrotatedSimple>
class Simple {
 protected:
  using Base = UnrotatedSimple;
  static constexpr int K = Base::kComponents;
  static constexpr int D = Base::kDimensions;

 public:
  static constexpr int kComponents = Base::kComponents;
  static constexpr int kDimensions = D;
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using MatKx1 = algebra::Matrix<Scalar, K, 1>;
  using Conservative = MatKx1;
  using Flux = MatKx1;
  using FluxMatrix = algebra::Matrix<Scalar, K, D>;
  using Frame3d = std::array<Vector, 3>;
  using Jacobi = typename Base::Jacobi;
  using Coefficient = typename Base::Coefficient;

 protected:
  template <class Value>
  static Flux ConvertToFlux(const Value& v) {
    Flux flux;
    std::memcpy(flux.data(), &v, K * sizeof(flux[0]));
    return flux;
  }

 public:
  void Rotate(Scalar n_x, Scalar n_y) {
    static_assert(D == 2);
    auto a_normal = convection_coefficient_[X] * n_x;
    a_normal += convection_coefficient_[Y] * n_y;
    unrotated_simple_ = UnrotatedSimple(a_normal);
  }
  void Rotate(Scalar n_x, Scalar n_y,  Scalar n_z) {
    static_assert(D == 3);
    Jacobi a_normal = convection_coefficient_[X] * n_x;
    a_normal += convection_coefficient_[Y] * n_y;
    a_normal += convection_coefficient_[Z] * n_z;
    unrotated_simple_ = UnrotatedSimple(a_normal);
  }
  void Rotate(const Frame3d &frame) {
    const auto &nu = frame[0];
    assert(std::abs(1 - nu.norm()) < 1e-6);
    Rotate(nu[X], nu[Y], nu[Z]);
  }
  Flux GetFluxUpwind(const Conservative& left,
      const Conservative& right) const {
    auto raw_flux = unrotated_simple_.GetFluxUpwind(left, right);
    return ConvertToFlux(raw_flux);
  }
  Flux GetFluxOnSolidWall(const Conservative& state) const {
    Flux flux;
    flux.setZero();
    return flux;
  }
  Flux GetFluxOnSupersonicOutlet(const Conservative& state) const {
    auto raw_flux = unrotated_simple_.GetFlux(state);
    return ConvertToFlux(raw_flux);
  }
  Flux GetFluxOnSupersonicInlet(const Conservative& state) const {
    auto raw_flux = unrotated_simple_.GetFlux(state);
    return ConvertToFlux(raw_flux);
  }
  Flux GetFluxOnSubsonicInlet(Conservative const& conservative_i,
      Conservative const& conservative_o) const {
    return {};
  }
  Flux GetFluxOnSubsonicOutlet(Conservative const& conservative_i,
      Conservative const& conservative_o) const {
    return {};
  }
  Flux GetFluxOnSmartBoundary(Conservative const& conservative_i,
      Conservative const& conservative_o) const {
    // TODO(PVC): provide physical implementation
    return {};
  }
  static FluxMatrix GetFluxMatrix(const Conservative& state) {
    FluxMatrix flux_mat;
    flux_mat.col(X) = convection_coefficient_[X] * state;
    flux_mat.col(Y) = convection_coefficient_[Y] * state;
    flux_mat.col(Z) = convection_coefficient_[Z] * state;
    return flux_mat;
  }
  static void SetConvectionCoefficient(Jacobi const &a_x, Jacobi const &a_y,
      Jacobi const &a_z) {
    convection_coefficient_[X] = a_x;
    convection_coefficient_[Y] = a_y;
    convection_coefficient_[Z] = a_z;
  }

 protected:
  UnrotatedSimple unrotated_simple_;
  static Coefficient convection_coefficient_;
};
template <class UnrotatedSimple>
typename Simple<UnrotatedSimple>::Coefficient
Simple<UnrotatedSimple>::convection_coefficient_;

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_ROTATED_SIMPLE_HPP_
