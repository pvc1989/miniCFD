// Copyright 2023 PEI Weicheng
#ifndef MINI_RIEMANN_ROTATED_DIRECT_DG_HPP_
#define MINI_RIEMANN_ROTATED_DIRECT_DG_HPP_

namespace mini {
namespace riemann {
namespace rotated {

template <typename Scalar>
class DirectDG {
  Scalar const beta_0_;
  Scalar const beta_1_;
  static constexpr int X{0};
  static constexpr int Y{1};
  static constexpr int Z{2};
  static constexpr int XX{0};
  static constexpr int XY{1}; static constexpr int YX{XY};
  static constexpr int XZ{2}; static constexpr int ZX{XZ};
  static constexpr int YY{3};
  static constexpr int YZ{4}; static constexpr int ZY{YZ};
  static constexpr int ZZ{5};

 public:
  DirectDG(Scalar beta_0 = 2.0, Scalar beta_1 = 1.0 / 12)
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

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_ROTATED_DIRECT_DG_HPP_
