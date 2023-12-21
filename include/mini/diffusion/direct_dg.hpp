// Copyright 2023 PEI Weicheng
#ifndef MINI_DIFFUSION_DIRECT_DG_HPP_
#define MINI_DIFFUSION_DIRECT_DG_HPP_

#include "mini/constant/index.hpp"

namespace mini {
namespace diffusion {

template <typename Scalar>
class DirectDG {
  Scalar const beta_0_;
  Scalar const beta_1_;
  using namespace mini::contant::index;

 public:
  explicit DirectDG(Scalar beta_0 = 2.0, Scalar beta_1 = 1.0 / 12)
      : beta_0_(beta_0), beta_1_(beta_1) {}

  template <typename Value, typename Gradient, typename Hessian>
  Gradient GetCommonGradient(Scalar distance, Normal normal,
      Value const &left_value, Value const &right_value,
      Gradient const &left_gradient, Gradient const &right_gradient,
      Hessian const &left_hessian, Hessian const &right_hessian) const {
    // add the average of Gradient
    Gradient common_gradient = (left_gradient + right_gradient) / 2;
    // add the penalty of Value jump
    Scalar factor_0 = beta_0_ / distance;
    normal *= factor_0;
    Value value_jump = right_value - left_value;
    common_gradient.row(X) += normal[X] * value_jump;
    common_gradient.row(Y) += normal[Y] * value_jump;
    common_gradient.row(Z) += normal[Z] * value_jump;
    // add the penalty of Hessian jump
    normal *= beta_1_ * distance / factor_0;
    Value hessian_jump = right_hessian - left_hessian;
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

}  // namespace diffusion
}  // namespace mini

#endif  // MINI_DIFFUSION_DIRECT_DG_HPP_
