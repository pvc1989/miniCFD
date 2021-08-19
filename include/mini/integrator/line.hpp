//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_INTEGRATOR_LINE_HPP_
#define MINI_INTEGRATOR_LINE_HPP_

#include <Eigen/Dense>

#include <type_traits>

template <typename Scalar = double, int Q = 4>
struct GaussIntegrator {
  static const std::array<Scalar, Q> points;
  static const std::array<Scalar, Q> weights;
};

template <typename Scalar>
struct GaussIntegrator<Scalar, 4> {
  static const std::array<Scalar, 4> points;
  static const std::array<Scalar, 4> weights;
};
template <typename Scalar>
std::array<Scalar, 4> const GaussIntegrator<Scalar, 4>::points = {
  (Scalar) -std::sqrt((3 - 2 * std::sqrt(1.2)) / 7),
  (Scalar) +std::sqrt((3 - 2 * std::sqrt(1.2)) / 7),
  (Scalar) -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7),
  (Scalar) +std::sqrt((3 + 2 * std::sqrt(1.2)) / 7),
};
template <typename Scalar>
std::array<Scalar, 4> const GaussIntegrator<Scalar, 4>::weights = {
  (Scalar) (18 + std::sqrt(30)) / 36.0,
  (Scalar) (18 + std::sqrt(30)) / 36.0,
  (Scalar) (18 - std::sqrt(30)) / 36.0,
  (Scalar) (18 - std::sqrt(30)) / 36.0
};

template <typename Scalar = double, int Q = 4, /* dim(space) */int D = 1>
class Line {
  using Mat2x1 = Eigen::Matrix<Scalar, 2, 1>;
  using MatDx1 = Eigen::Matrix<Scalar, D, 1>;
  using MatDx2 = Eigen::Matrix<Scalar, D, 2>;
  using Arr1x2 = Eigen::Array<Scalar, 1, 2>;
  using Integrator = GaussIntegrator<Scalar, Q>;

 public:
  static const Arr1x2 x_local_i_;
  MatDx2 xyz_global_Dx2_;
  static const Mat2x1 diff_shape_2x1_;

  static Mat2x1 shape_2x1(Scalar x_local) {
    Arr1x2 n_1x2;
    n_1x2  = (1 + x_local_i_ * x_local);
    n_1x2 /= 2;
    return n_1x2.transpose();
  }
  static Mat2x1 diff_shape_local_2x1(Scalar x_local) {
    return diff_shape_2x1_;
  }
  MatDx1 jacobian(Scalar x_local) {
    return xyz_global_Dx2_ * diff_shape_local_2x1(x_local);
  }
  template <typename Callable>
  auto gauss_quadrature(Callable&& f_in_local) {
    auto sum = f_in_local(0);
    sum *= 0;
    for (int i = 0; i < Q; ++i) {
      Scalar weight = Integrator::weights[i];
      Scalar x_local = Integrator::points[i];
      auto f_val = f_in_local(x_local);
      f_val *= weight;
      sum += f_val;
    }
    return sum;
  }


 public:
  explicit Line(MatDx2 const& x_global_i) {
    xyz_global_Dx2_ = x_global_i;
  }
  MatDx1 local_to_global_Dx1(Scalar x_local) {
    return xyz_global_Dx2_ * shape_2x1(x_local);
  }
  Scalar global_to_local_Dx1(MatDx1 xyz_global) {
    return 0;
  }
  template <typename Callable>
  auto integrate(Callable&& f_in_global) {
    auto f_in_local = [this, &f_in_global](Scalar x_local) {
      auto xyz_global = local_to_global_Dx1(x_local);
      auto f_val = f_in_global(xyz_global);
      auto mat_j = jacobian(x_local);
      auto det_j = mat_j.norm();
      f_val *= det_j;
      return f_val;
    };
    return gauss_quadrature(f_in_local);
  }
  template <typename Callable>
  Scalar innerprod(Callable&& f, Callable&& g) {
    return integrate([&f, &g](Scalar x){
      return f(x) * g(x);
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
    auto A = integrate([this, &raw_basis](MatDx1 xyz){
      auto col = raw_basis(xyz);
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

template <typename Scalar, int Q, int D>
typename Line<Scalar, Q, D>::Arr1x2 const
Line<Scalar, Q, D>::x_local_i_ = {-1, +1};

template <typename Scalar, int Q, int D>
typename Line<Scalar, Q, D>::Mat2x1 const
Line<Scalar, Q, D>::diff_shape_2x1_ = {-0.5, 0.5};

#endif  // MINI_INTEGRATOR_LINE_HPP_
