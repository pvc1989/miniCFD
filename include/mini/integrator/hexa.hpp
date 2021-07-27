
#include <iostream>
#include <type_traits>
#include <Eigen/Dense>

#include "line.h"

using std::cout;
using std::endl;

template <typename Scalar = double, int Q1d = 4>
class Hexa {
  using Mat3x3 = Eigen::Matrix<Scalar, 3, 3>;
  using Mat1x8 = Eigen::Matrix<Scalar, 1, 8>;
  using Mat8x1 = Eigen::Matrix<Scalar, 8, 1>;
  using Mat3x8 = Eigen::Matrix<Scalar, 3, 8>;
  using Mat8x3 = Eigen::Matrix<Scalar, 8, 3>;
  using Mat3x1 = Eigen::Matrix<Scalar, 3, 1>;
  using Mat10x1 = Eigen::Matrix<Scalar, 10, 1>;

  using Arr1x8 = Eigen::Array<Scalar, 1, 8>;
  using Arr8x1 = Eigen::Array<Scalar, 8, 1>;
  using Arr3x8 = Eigen::Array<Scalar, 3, 8>;
  using Arr8x3 = Eigen::Array<Scalar, 8, 3>;

  using Integrator = GaussIntegrator<Scalar, Q1d>;

 private:
  static const Arr1x8 x_local_i_;
  static const Arr1x8 y_local_i_;
  static const Arr1x8 z_local_i_;
  Mat3x8 xyz_global_3x8_;

  static Mat8x1 shape_8x1(Scalar x_local, Scalar y_local, Scalar z_local) {
    Arr1x8 n_1x8;
    n_1x8  = (1 + x_local_i_ * x_local);
    n_1x8 *= (1 + y_local_i_ * y_local);
    n_1x8 *= (1 + z_local_i_ * z_local);
    n_1x8 /= 8;
    return n_1x8.transpose();
  }
  static Mat8x3 diff_shape_local_8x3(Scalar x_local, Scalar y_local, Scalar z_local) { 
    Arr8x3 dn;
    Arr8x1 factor_x = x_local_i_.transpose() * x_local; factor_x += 1;
    Arr8x1 factor_y = y_local_i_.transpose() * y_local; factor_y += 1;
    Arr8x1 factor_z = z_local_i_.transpose() * z_local; factor_z += 1;
    dn.col(0) << x_local_i_.transpose() * factor_y * factor_z;
    dn.col(1) << y_local_i_.transpose() * factor_x * factor_z;
    dn.col(2) << z_local_i_.transpose() * factor_x * factor_y;
    dn /= 8;
    return dn;
  }
  Mat3x3 jacobian(Scalar x_local, Scalar y_local, Scalar z_local) {
    return xyz_global_3x8_ * diff_shape_local_8x3(x_local, y_local, z_local);
  }
  Mat3x3 jacobian(Mat3x1 xyz_local) {
    return jacobian(xyz_local[0], xyz_local[1], xyz_local[2]);
  }
  template <typename Callable, typename MatJ>
  static Mat3x1 root(Callable&& func, Mat3x1 x, MatJ&& matj, Scalar xtol = 1e-5) {
    Mat3x1 res;
    do {
      res = matj(x).partialPivLu().solve(func(x));
      x -= res;
    } while (res.norm() > xtol);
    return x;
  }
  template <typename Callable>
  auto gauss_quadrature(Callable&& f_in_local) {
    auto sum = f_in_local(0, 0, 0);
    sum *= 0;
    for (int i = 0; i < 4; ++i) {
      Scalar x_weight = Integrator::weights[i];
      Scalar x_local = Integrator::points[i];
      for (int j = 0; j < 4; ++j) {
        Scalar xy_weight = x_weight * Integrator::weights[j];
        Scalar y_local = Integrator::points[j];
        for (int k = 0; k < 4; ++k) {
          Scalar xyz_weight = xy_weight * Integrator::weights[k];
          Scalar z_local = Integrator::points[k];
          auto f_val = f_in_local(x_local, y_local, z_local);
          f_val *= xyz_weight;
          sum += f_val;
        }
      }
    }
    return sum;
  }
 public:
  Hexa(Mat1x8 const& x_global_i, Mat1x8 const& y_global_i, Mat1x8 const& z_global_i) {
    xyz_global_3x8_.row(0) << x_global_i;
    xyz_global_3x8_.row(1) << y_global_i;
    xyz_global_3x8_.row(2) << z_global_i;
    cout << xyz_global_3x8_ << endl;
  }
  Mat3x1 local_to_global_3x1(Scalar x_local, Scalar y_local, Scalar z_local) {
    return xyz_global_3x8_ * shape_8x1(x_local, y_local, z_local);
  }
  Mat3x1 local_to_global_3x1(Mat3x1 xyz_local) {
    return local_to_global_3x1(xyz_local[0], xyz_local[1], xyz_local[2]);
  }
  Mat3x1 global_to_local_3x1(Scalar x_global, Scalar y_global, Scalar z_global) {
    Mat3x1 xyz_global = {x_global, y_global, z_global};
    auto func = [this, &xyz_global](Mat3x1 const& xyz_local) {
      auto res = local_to_global_3x1(xyz_local);
      return res -= xyz_global;
    };
    auto jac = [this](Mat3x1 const& xyz_local) {
      return jacobian(xyz_local);
    };
    Mat3x1 xyz0 = {0, 0, 0};
    return root(func, xyz0, jac);
  }
  Mat3x1 global_to_local_3x1(Mat3x1 xyz_global) {
    return global_to_local_3x1(xyz_global[0], xyz_global[1], xyz_global[2]);
  }
  template <typename Callable>
  auto integrate(Callable&& f_in_global) {
    auto f_in_local = [this, &f_in_global](Scalar x_local, Scalar y_local, Scalar z_local) {
      auto xyz_global = local_to_global_3x1(x_local, y_local, z_local);
      auto f_val = f_in_global(xyz_global[0], xyz_global[1], xyz_global[2]);
      auto mat_j = jacobian(x_local, y_local, z_local);
      auto det_j = mat_j.determinant();
      f_val *= det_j;
      return f_val;
    };
    return gauss_quadrature(f_in_local);
  }
  template <typename Callable>
  Scalar innerprod(Callable&& f, Callable&& g) {
    return integrate([&f, &g](Scalar x, Scalar y, Scalar z){
      return f(x, y, z) * g(x, y, z);
    });
  }
  template <typename Callable>
  Scalar norm(Callable&& f) {
    return std::sqrt(innerprod(f, f));
  }
  template <int N, typename Callable>
  auto orthonormalize(Callable&& raw_basis) {
    using MatNxN = Eigen::Matrix<Scalar, N, N>;
    MatNxN S;
    S.setIdentity();
    auto A = integrate([this, &raw_basis](Scalar x, Scalar y, Scalar z){
      auto col = raw_basis(x, y, z);
      MatNxN result = col * col.transpose();
      return result;
    });
    S(0, 0) = 1 / std::sqrt(A(0, 0));
    for (int i = 1; i < N; ++i) {
      for (int j = 0; j < i; ++j) {
        Scalar temp = 0;
        for (int k = 0; k <= j; ++k) {
          temp += S(j, k) * A(k, i);
        }
        for (int l = 0; l <= j; ++l) {
          S(i, l) -= temp * S(j, l);
        }
      }
      Scalar norm_sq = 0;
      for (int j = 0; j <= i; ++j) {
        Scalar sum = 0, Sij = S(i, j);
        for (int k = 0; k < j; ++k) {
          sum += 2 * S(i, k) * A(k, j);
        }
        norm_sq += Sij * (Sij * A(j, j) + sum);
      }
      S.row(i) /= std::sqrt(norm_sq);
    }
    return S;
  }
};
template <typename Scalar, int Q1d>
typename Hexa<Scalar, Q1d>::Arr1x8 const Hexa<Scalar, Q1d>::x_local_i_ = {-1, +1, +1, -1, -1, +1, +1, -1};
template <typename Scalar, int Q1d>
typename Hexa<Scalar, Q1d>::Arr1x8 const Hexa<Scalar, Q1d>::y_local_i_ = {-1, -1, +1, +1, -1, -1, +1, +1};
template <typename Scalar, int Q1d>
typename Hexa<Scalar, Q1d>::Arr1x8 const Hexa<Scalar, Q1d>::z_local_i_ = {-1, -1, -1, -1, +1, +1, +1, +1};
