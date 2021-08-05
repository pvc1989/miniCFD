//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_INTEGRATOR_QUAD_HPP_
#define MINI_INTEGRATOR_QUAD_HPP_

#include <Eigen/Dense>

#include "mini/integrator/line.hpp"

template <typename Scalar = double, int Q1d = 4, int D = 2>
class Quad {
  using Arr1x4 = Eigen::Array<Scalar, 1, 4>;
  using Arr4x1 = Eigen::Array<Scalar, 4, 1>;
  using Arr4x2 = Eigen::Array<Scalar, 4, 2>;

  using MatDx4 = Eigen::Matrix<Scalar, D, 4>;
  using MatDx2 = Eigen::Matrix<Scalar, D, 2>;
  using MatDx1 = Eigen::Matrix<Scalar, D, 1>;
  using Mat4x1 = Eigen::Matrix<Scalar, 4, 1>;
  using Mat4x2 = Eigen::Matrix<Scalar, 4, 2>;
  using Mat2x1 = Eigen::Matrix<Scalar, 2, 1>;

  using Integrator = GaussIntegrator<Scalar, Q1d>;

 public:
  static const Arr1x4 x_local_i_;
  static const Arr1x4 y_local_i_;
  MatDx4 xyz_global_Dx4_;

  static Mat4x1 shape_4x1(Mat2x1 xy_local) {
    Arr1x4 n_1x4;
    n_1x4  = (1 + x_local_i_ * xy_local[0]);
    n_1x4 *= (1 + y_local_i_ * xy_local[1]);
    n_1x4 /= 4.0;
    return n_1x4.transpose();
  }
  static Mat4x1 shape_4x1(Scalar x, Scalar y) {
    Arr1x4 n_1x4;
    n_1x4  = (1 + x_local_i_ * x);
    n_1x4 *= (1 + y_local_i_ * y);
    n_1x4 /= 4.0;
    return n_1x4.transpose();
  }
  static Mat4x2 diff_shape_local_4x2(Scalar x_local, Scalar y_local) {
    Arr4x2 dn;
    Arr4x1 factor_x = x_local_i_.transpose() * x_local; factor_x += 1;
    Arr4x1 factor_y = y_local_i_.transpose() * y_local; factor_y += 1;
    dn.col(0) << x_local_i_.transpose() * factor_y;
    dn.col(1) << y_local_i_.transpose() * factor_x;
    dn /= 4.0;
    return dn;
  }
  MatDx2 jacobian(Scalar x_local, Scalar y_local) {
    return xyz_global_Dx4_ * diff_shape_local_4x2(x_local, y_local);
  }
  template <typename Callable>
  auto gauss_quadrature(Callable&& f_in_local) {
    auto sum = f_in_local(0, 0);
    sum *= 0;
    for (int i = 0; i < Q1d; ++i) {
      Scalar x_weight = Integrator::weights[i];
      Scalar x_local = Integrator::points[i];
      for (int j = 0; j < Q1d; ++j) {
        Scalar y_weight = Integrator::weights[j];
        Scalar y_local = Integrator::points[j];
        auto f_val = f_in_local(x_local, y_local);
        f_val *= x_weight * y_weight;
        sum += f_val;
      }
    }
    return sum;
  }

 public:
  explicit Quad(MatDx4 xyz_global) {
    xyz_global_Dx4_ = xyz_global;
    print(xyz_global_Dx4_);
  }
  MatDx1 local_to_global_Dx1(Mat2x1 xy_local) {
    return xyz_global_Dx4_ * shape_4x1(xy_local);
  }
  MatDx1 local_to_global_Dx1(Scalar x, Scalar y) {
    return xyz_global_Dx4_ * shape_4x1(x, y);
  }
  template <typename Callable>
  auto integrate(Callable&& f_in_global) {
    auto f_in_local = [this, &f_in_global](Scalar x_local, Scalar y_local) {
      auto xyz_global = local_to_global_Dx1(x_local, y_local);
      auto f_val = f_in_global(xyz_global);
      auto mat_j = jacobian(x_local, y_local);
      auto det_j = (mat_j.transpose() * mat_j).determinant();
      f_val *= std::sqrt(det_j);
      return f_val;
    };
    return gauss_quadrature(f_in_local);
  }
};

template <typename Scalar, int Q1d, int D>
typename Quad<Scalar, Q1d, D>::Arr1x4 const
Quad<Scalar, Q1d, D>::x_local_i_ = {-1, +1, +1, -1};

template <typename Scalar, int Q1d, int D>
typename Quad<Scalar, Q1d, D>::Arr1x4 const
Quad<Scalar, Q1d, D>::y_local_i_ = {-1, -1, +1, +1};

#endif  // MINI_INTEGRATOR_QUAD_HPP_
