// Copyright 2023 PEI Weicheng
#ifndef MINI_RIEMANN_DIFFUSIVE_DIRECT_DG_HPP_
#define MINI_RIEMANN_DIFFUSIVE_DIRECT_DG_HPP_

#include "mini/constant/index.hpp"

namespace mini {
namespace riemann {
namespace diffusive {

template <typename DiffusionModel>
class DirectDG : public Model {
  Scalar const beta_0_ = 2.0;
  Scalar const beta_1_ = 1.0 / 12;
  using namespace mini::contant::index;
  using Base = DiffusionModel;

 public:
  using Scalar = typename Base::Scalar;
  using Conservative = typename Base::Conservative;
  using Gradient = typename Base::Gradient;
  using FluxMatrix = typename Base::FluxMatrix;
  using Flux = typename Base::Flux;

  Gradient GetCommonGradient(Scalar distance, Normal normal,
      Conservative const &left_value, Conservative const &right_value,
      Gradient const &left_gradient, Gradient const &right_gradient) const {
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

  template <typename Hessian>
  Gradient GetCommonGradient(Scalar distance, Normal normal,
      Conservative const &left_value, Conservative const &right_value,
      Gradient const &left_gradient, Gradient const &right_gradient,
      Hessian const &left_hessian, Hessian const &right_hessian) const {
    Gradient common_gradient = GetCommonGradient(distance, normal,
      left_value, right_value, left_gradient, right_gradient);
    // add the penalty of Hessian jump
    normal *= beta_1_ * distance;
    Hessian hessian_jump = right_hessian - left_hessian;
    common_gradient.row(X) += normal[X] * hessian_jump[XX];
    common_gradient.row(X) += normal[Y] * hessian_jump[XY];
    common_gradient.row(X) += normal[Z] * hessian_jump[XZ];
    common_gradient.row(Y) += normal[X] * hessian_jump[YX];
    common_gradient.row(Y) += normal[Y] * hessian_jump[YY];
    common_gradient.row(Y) += normal[Z] * hessian_jump[YZ];
    common_gradient.row(Z) += normal[X] * hessian_jump[ZX];
    common_gradient.row(Z) += normal[Y] * hessian_jump[ZY];
    common_gradient.row(Z) += normal[Z] * hessian_jump[ZZ];
    return common_gradient;
  }
};

}  // namespace diffusive
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_DIFFUSIVE_DIRECT_DG_HPP_
