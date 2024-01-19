// Copyright 2023 PEI Weicheng
#ifndef MINI_RIEMANN_DIFFUSIVE_DIRECT_DG_HPP_
#define MINI_RIEMANN_DIFFUSIVE_DIRECT_DG_HPP_

#include "mini/algebra/eigen.hpp"
#include "mini/constant/index.hpp"

namespace mini {
namespace riemann {
namespace diffusive {

using namespace mini::constant::index;

template <typename DiffusionModel>
class DirectDG : public DiffusionModel {
  using Base = DiffusionModel;

 public:
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using Conservative = typename Base::Conservative;
  using Gradient = typename Base::Gradient;
  using FluxMatrix = typename Base::FluxMatrix;
  using Flux = typename Base::Flux;
  using Hessian = algebra::Matrix<Scalar, 6, Base::kComponents>;

 protected:
  static Scalar beta_0_;
  static Scalar beta_1_;

 public:
  static Gradient GetCommonGradient(Scalar distance, Vector normal,
      Conservative const &left_value, Conservative const &right_value,
      Gradient const &left_gradient, Gradient const &right_gradient) {
    // add the average of Gradient
    Gradient common_gradient = (left_gradient + right_gradient) / 2;
    // add the penalty of Value jump
    normal *= beta_0_ / distance;
    Conservative value_jump = right_value - left_value;
    common_gradient.row(X) += normal[X] * value_jump;
    common_gradient.row(Y) += normal[Y] * value_jump;
    common_gradient.row(Z) += normal[Z] * value_jump;
    return common_gradient;
  }

  static Gradient GetCommonGradient(Scalar distance, Vector normal,
      Conservative const &left_value, Conservative const &right_value,
      Gradient const &left_gradient, Gradient const &right_gradient,
      Hessian const &left_hessian, Hessian const &right_hessian) {
    Gradient common_gradient = GetCommonGradient(distance, normal,
        left_value, right_value, left_gradient, right_gradient);
    // add the penalty of Hessian jump
    normal *= beta_1_ * distance;
    Hessian hessian_jump = right_hessian - left_hessian;
    common_gradient.row(X) += normal[X] * hessian_jump.row(XX);
    common_gradient.row(X) += normal[Y] * hessian_jump.row(XY);
    common_gradient.row(X) += normal[Z] * hessian_jump.row(XZ);
    common_gradient.row(Y) += normal[X] * hessian_jump.row(YX);
    common_gradient.row(Y) += normal[Y] * hessian_jump.row(YY);
    common_gradient.row(Y) += normal[Z] * hessian_jump.row(YZ);
    common_gradient.row(Z) += normal[X] * hessian_jump.row(ZX);
    common_gradient.row(Z) += normal[Y] * hessian_jump.row(ZY);
    common_gradient.row(Z) += normal[Z] * hessian_jump.row(ZZ);
    return common_gradient;
  }

  static void SetBetaValues(Scalar beta_0, Scalar beta_1) {
    beta_0_ = beta_0;
    beta_1_ = beta_1;
  }
};
template <typename DiffusionModel>
typename DirectDG<DiffusionModel>::Scalar DirectDG<DiffusionModel>::beta_0_;
template <typename DiffusionModel>
typename DirectDG<DiffusionModel>::Scalar DirectDG<DiffusionModel>::beta_1_;

}  // namespace diffusive
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_DIFFUSIVE_DIRECT_DG_HPP_
