//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_INTEGRATOR_LINE_HPP_
#define MINI_INTEGRATOR_LINE_HPP_

#include <cmath>
#include <type_traits>

#include "mini/algebra/eigen.hpp"
#include "mini/integrator/function.hpp"
#include "mini/integrator/basis.hpp"

namespace mini {
namespace integrator {

template <typename Scalar = double, int Q = 4>
struct GaussLegendre;

template <typename Scalar>
struct GaussLegendre<Scalar, 1> {
  using Mat1x1 = algebra::Matrix<Scalar, 1, 1>;
  static const Mat1x1 points;
  static const Mat1x1 weights;
  static Mat1x1 BuildPoints() {
    return { 0.0 };
  }
  static Mat1x1 BuildWeights() {
    return { 2.0 };
  }
};
template <typename Scalar>
typename GaussLegendre<Scalar, 1>::Mat1x1 const
GaussLegendre<Scalar, 1>::points =
    GaussLegendre<Scalar, 1>::BuildPoints();
template <typename Scalar>
typename GaussLegendre<Scalar, 1>::Mat1x1 const
GaussLegendre<Scalar, 1>::weights =
    GaussLegendre<Scalar, 1>::BuildWeights();

template <typename Scalar>
struct GaussLegendre<Scalar, 2> {
  using Mat1x2 = algebra::Matrix<Scalar, 1, 2>;
  static const Mat1x2 points;
  static const Mat1x2 weights;
  static Mat1x2 BuildPoints() {
    return { -std::sqrt(1.0/3.0), +std::sqrt(1.0/3.0) };
  }
  static Mat1x2 BuildWeights() {
    return { 1.0, 1.0 };
  }
};
template <typename Scalar>
typename GaussLegendre<Scalar, 2>::Mat1x2 const
GaussLegendre<Scalar, 2>::points =
    GaussLegendre<Scalar, 2>::BuildPoints();
template <typename Scalar>
typename GaussLegendre<Scalar, 2>::Mat1x2 const
GaussLegendre<Scalar, 2>::weights =
    GaussLegendre<Scalar, 2>::BuildWeights();

template <typename Scalar>
struct GaussLegendre<Scalar, 3> {
  using Mat1x3 = algebra::Matrix<Scalar, 1, 3>;
  static const Mat1x3 points;
  static const Mat1x3 weights;
  static Mat1x3 BuildPoints() {
    return { -std::sqrt(0.6), 0.0, +std::sqrt(0.6) };
  }
  static Mat1x3 BuildWeights() {
    return { 5.0/9.0, 8.0/9.0, 5.0/9.0 };
  }
};
template <typename Scalar>
typename GaussLegendre<Scalar, 3>::Mat1x3 const
GaussLegendre<Scalar, 3>::points =
    GaussLegendre<Scalar, 3>::BuildPoints();
template <typename Scalar>
typename GaussLegendre<Scalar, 3>::Mat1x3 const
GaussLegendre<Scalar, 3>::weights =
    GaussLegendre<Scalar, 3>::BuildWeights();

template <typename Scalar>
struct GaussLegendre<Scalar, 4> {
  using Mat1x4 = algebra::Matrix<Scalar, 1, 4>;
  static const Mat1x4 points;
  static const Mat1x4 weights;
  static Mat1x4 BuildPoints() {
    return {
        -std::sqrt((3 - 2 * std::sqrt(1.2)) / 7),
        +std::sqrt((3 - 2 * std::sqrt(1.2)) / 7),
        -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7),
        +std::sqrt((3 + 2 * std::sqrt(1.2)) / 7),
    };
  }
  static Mat1x4 BuildWeights() {
    return {
        (18 + std::sqrt(30)) / 36,
        (18 + std::sqrt(30)) / 36,
        (18 - std::sqrt(30)) / 36,
        (18 - std::sqrt(30)) / 36,
    };
  }
};
template <typename Scalar>
typename GaussLegendre<Scalar, 4>::Mat1x4 const
GaussLegendre<Scalar, 4>::points =
    GaussLegendre<Scalar, 4>::BuildPoints();
template <typename Scalar>
typename GaussLegendre<Scalar, 4>::Mat1x4 const
GaussLegendre<Scalar, 4>::weights =
    GaussLegendre<Scalar, 4>::BuildWeights();

template <typename Scalar>
struct GaussLegendre<Scalar, 5> {
  using Mat1x5 = algebra::Matrix<Scalar, 1, 5>;
  static const Mat1x5 points;
  static const Mat1x5 weights;
  static Mat1x5 BuildPoints() {
    return {
        -std::sqrt((5 - std::sqrt(40 / 7.0)) / 9),
        +std::sqrt((5 - std::sqrt(40 / 7.0)) / 9),
        0,
        -std::sqrt((5 + std::sqrt(40 / 7.0)) / 9),
        +std::sqrt((5 + std::sqrt(40 / 7.0)) / 9),
    };
  }
  static Mat1x5 BuildWeights() {
    return {
        (322 + 13 * std::sqrt(70.0)) / 900,
        (322 + 13 * std::sqrt(70.0)) / 900,
        128.0 / 225.0,
        (322 - 13 * std::sqrt(70.0)) / 900,
        (322 - 13 * std::sqrt(70.0)) / 900,
    };
  }
};
template <typename Scalar>
typename GaussLegendre<Scalar, 5>::Mat1x5 const
GaussLegendre<Scalar, 5>::points =
    GaussLegendre<Scalar, 5>::BuildPoints();
template <typename Scalar>
typename GaussLegendre<Scalar, 5>::Mat1x5 const
GaussLegendre<Scalar, 5>::weights =
    GaussLegendre<Scalar, 5>::BuildWeights();

template <typename Scalar = double, int Q = 4, /* dim(space) */int D = 1>
class Line {
  using Mat2x1 = algebra::Matrix<Scalar, 2, 1>;
  using MatDx1 = algebra::Matrix<Scalar, D, 1>;
  using MatDx2 = algebra::Matrix<Scalar, D, 2>;
  using Arr1x2 = algebra::Array<Scalar, 1, 2>;
  using Integrator = GaussLegendre<Scalar, Q>;

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
  MatDx1 Jacobian(Scalar x_local) {
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
  MatDx1 LocalToGlobal(Scalar x_local) {
    return xyz_global_Dx2_ * shape_2x1(x_local);
  }
  Scalar global_to_local_Dx1(MatDx1 xyz_global) {
    return 0;
  }
  template <typename Callable>
  auto integrate(Callable&& f_in_global) {
    auto f_in_local = [this, &f_in_global](Scalar x_local) {
      auto xyz_global = LocalToGlobal(x_local);
      auto f_val = f_in_global(xyz_global);
      auto mat_j = Jacobian(x_local);
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
    using MatNxN = algebra::Matrix<Scalar, N, N>;
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

}  // namespace integrator
}  // namespace mini

#endif  // MINI_INTEGRATOR_LINE_HPP_
