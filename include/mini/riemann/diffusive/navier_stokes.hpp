// Copyright 2023 PEI Weicheng
#ifndef MINI_RIEMANN_DIFFUSIVE_NAVIER_STOKES_HPP_
#define MINI_RIEMANN_DIFFUSIVE_NAVIER_STOKES_HPP_

#include "mini/algebra/eigen.hpp"
#include "mini/riemann/euler.hpp"
#include "mini/constant/index.hpp"

namespace mini {
namespace riemann {
namespace diffusive {

template <typename G>
class NavierStokes {
 public:
  static constexpr int kDimensions = 3;
  static constexpr int kComponents = 5;
  using Gas = G
  using Scalar = typename Gas::Scalar;
  using Gradient = algebra::Matrix<Scalar, kDimensions, kComponents>;
  using FluxMatrix = algebra::Matrix<Scalar, kComponents, kDimensions>;
  using Flux = euler::FluxTuple<Scalar, kDimensions>;
  using Conservative = euler::Conservatives<Scalar, kComponents>;
  using Primitive = euler::Primitive<Scalar, kComponents>;
  using Vector = typename Primitive::Vector;

  static void SetProperty(Scalar R) {
    R_ = R;
  }

 protected:
  static Scalar R_;

  static Gradient ConservativeGradientToPrimitiveGradient(
      Conservative const &c_val, Gradient const &c_grad) {
    auto p_val = ConservativeToPrimitive(c_val);
    Gradient p_grad;
    using namespace mini::constant::index;
    auto &grad_u = p_grad.col(X);
    auto &grad_v = p_grad.col(Y);
    auto &grad_w = p_grad.col(Z);
    auto &grad_rho = c_grad.col(0);
    auto &grad_rho_u = c_grad.col(1 + X);
    auto &grad_rho_v = c_grad.col(1 + Y);
    auto &grad_rho_w = c_grad.col(1 + Z);
    auto [rho, u, v, w, p] = p_val;
    grad_u = (grad_rho_u - u * grad_rho) / rho;
    grad_v = (grad_rho_v - v * grad_rho) / rho;
    grad_w = (grad_rho_w - w * grad_rho) / rho;
    auto [rho_u, rho_v, rho_w] = c_val.momentum();
    auto &grad_p = p_grad.col(3);
    grad_p  = u * grad_rho_u + rho_u * grad_u;
    grad_p += v * grad_rho_v + rho_v * grad_v;
    grad_p += w * grad_rho_w + rho_w * grad_w;
    grad_p *= -0.5;
    grad_p += c_grad.col(4);
    grad_p *= Gas::GammaMinusOne();
    auto &grad_T = p_grad.col(4);
    grad_T = grad_p / rho - (p / (rho * rho)) * grad_rho;
    grad_T /= R_;
    return p_grad;
  }

 public:
  static void ModifyFluxMatrix(Conservative const &value, Gradient const &gradient,
      FluxMatrix *flux) {
    using namespace mini::constant::index;
    
  }

  static void ModifyCommonFlux(Conservative const &value, Gradient const &gradient,
      Vector const &normal, Flux *flux) {
  }
};

}  // namespace diffusive
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_DIFFUSIVE_NAVIER_STOKES_HPP_
