// Copyright 2019 PEI Weicheng and YANG Minghao
#ifndef MINI_RIEMANN_ROTATED_SIMPLE_HPP_
#define MINI_RIEMANN_ROTATED_SIMPLE_HPP_

#include <cstring>

#include "mini/algebra/eigen.hpp"

namespace mini {
namespace riemann {
namespace rotated {

template <class UnrotatedSimple>
class Simple {
 protected:
  using Base = UnrotatedSimple;
  static constexpr int K = Base::kComponents;
  static constexpr int D = Base::kDimensions;
  static constexpr int x{0}, y{1}, z{2};

 public:
  static constexpr int kComponents = Base::kComponents;
  static constexpr int kDimensions = D;
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using MatKx1 = algebra::Matrix<Scalar, K, 1>;
  using Conservative = MatKx1;
  using Flux = MatKx1;
  using FluxMatrix = algebra::Matrix<Scalar, K, D>;
  using Frame3d = mini::algebra::Matrix<Scalar, 3, 3>;
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
    auto a_normal = global_coefficient[x] * n_x;
    a_normal += global_coefficient[y] * n_y;
    unrotated_simple_ = UnrotatedSimple(a_normal);
  }
  void Rotate(Scalar n_x, Scalar n_y,  Scalar n_z) {
    static_assert(D == 3);
    Jacobi a_normal = global_coefficient[x] * n_x;
    a_normal += global_coefficient[y] * n_y;
    a_normal += global_coefficient[z] * n_z;
    unrotated_simple_ = UnrotatedSimple(a_normal);
  }
  void Rotate(const Frame3d &frame) {
    const auto &nu = frame.col(0);
    Rotate(nu[x], nu[y], nu[z]);
  }
  Flux GetFluxOnTimeAxis(const Conservative& left,
      const Conservative& right) const {
    auto raw_flux = unrotated_simple_.GetFluxOnTimeAxis(left, right);
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
    for (int c = 0; c < D; ++c) {
      flux_mat.col(c) = global_coefficient[c] * state;
    }
    return flux_mat;
  }

  static Coefficient global_coefficient;

 protected:
  UnrotatedSimple unrotated_simple_;
};
template <class UnrotatedSimple>
typename Simple<UnrotatedSimple>::Coefficient
Simple<UnrotatedSimple>::global_coefficient;

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_ROTATED_SIMPLE_HPP_
