// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_RIEMANN_LINEAR_DOUBLE_HPP_
#define MINI_RIEMANN_LINEAR_DOUBLE_HPP_

#include <cmath>

#include "mini/algebra/eigen.hpp"

namespace mini {
namespace riemann {
namespace linear {

template <int D>
class Double {
 public:
  static constexpr int kDim = D;
  static constexpr int kFunc = 2;
  using Scalar = double;
  using Vector = algebra::Vector<Scalar, kDim>;
  using Column = algebra::Vector<Scalar, kFunc>;
  using Matrix = algebra::Matrix<Scalar, kFunc, kFunc>;
  using Jacobi = Matrix;
  using Coefficient = algebra::Vector<Jacobi, kDim>;
  using Conservative = Column;
  using Flux = Column;
  // Constructor:
  Double() : a_const_{{1, 0}, {0, -1}} { Decompose(); }
  explicit Double(const Matrix &a_const) : a_const_(a_const) { Decompose(); }
  // Get F on T Axia
  Flux GetFluxOnTimeAxis(const Conservative &left, const Conservative &right)
      const {
    Flux flux;
    if (0 <= eigen_values_[0]) {
      flux = GetFlux(left);
    } else if (0 >= eigen_values_[1]) {
      flux = GetFlux(right);
    } else {
      flux = FluxInsideSector(left, right, 1);
    }
    return flux;
  }
  // Get F of U
  Flux GetFlux(const Conservative &state) const {
    return a_const_ * state;
  }

 private:
  Flux FluxInsideSector(const Conservative &left, const Conservative &right,
      int k) const {
    Flux flux{0, 0};
    using Row = Column;
    for (int i = 0; i < k; i++) {
      auto l = Row(eigen_matrix_l_(i, 0), eigen_matrix_l_(i, 1));
      double temp = l.dot(right) * eigen_values_[i];
      flux[0] += temp * eigen_matrix_r_(0, i);
      flux[1] += temp * eigen_matrix_r_(1, i);
    }
    for (int i = k; i < 2; i++) {
      auto l = Row(eigen_matrix_l_(i, 0), eigen_matrix_l_(i, 1));
      double temp = l.dot(left) * eigen_values_[i];
      flux[0] += temp * eigen_matrix_r_(0, i);
      flux[1] += temp * eigen_matrix_r_(1, i);
    }
    return flux;
  }
  void Decompose() {
    GetEigenValues();
    GetEigenVectors();
    GetInverseEigenVectors();
  }
  void GetEigenValues() {
    double b = a_const_(0, 0) + a_const_(1, 1);
    double c = a_const_(0, 0) * a_const_(1, 1) -
               a_const_(0, 1) * a_const_(1, 0);
    double delta = std::sqrt(b * b - 4 * c);
    eigen_values_[0] = (b - delta) / 2;
    eigen_values_[1] = (b + delta) / 2;
  }
  void GetEigenVectors() {
    double a = a_const_(0, 0) - eigen_values_[0];
    double b = a_const_(0, 1);
    double c = a_const_(1, 0);
    double d = a_const_(1, 1) - eigen_values_[0];
    if (a == 0 && b == 0) {
      eigen_matrix_r_(0, 0) = d;
      eigen_matrix_r_(1, 0) = -c;
    } else {
      eigen_matrix_r_(0, 0) = b;
      eigen_matrix_r_(1, 0) = -a;
    }
    a = a_const_(0, 0) - eigen_values_[1];
    d = a_const_(1, 1) - eigen_values_[1];
    if (a == 0 && b == 0) {
      eigen_matrix_r_(0, 1) = d;
      eigen_matrix_r_(1, 1) = -c;
    } else {
      eigen_matrix_r_(0, 1) = b;
      eigen_matrix_r_(1, 1) = -a;
    }
  }
  void GetInverseEigenVectors() {
    double det = eigen_matrix_r_(0, 0) * eigen_matrix_r_(1, 1) -
                 eigen_matrix_r_(0, 1) * eigen_matrix_r_(1, 0);
    eigen_matrix_l_(0, 0) =  eigen_matrix_r_(1, 1) / det;
    eigen_matrix_l_(0, 1) = -eigen_matrix_r_(0, 1) / det;
    eigen_matrix_l_(1, 0) = -eigen_matrix_r_(1, 0) / det;
    eigen_matrix_l_(1, 1) =  eigen_matrix_r_(0, 0) / det;
  }
  Matrix eigen_matrix_r_;
  Matrix eigen_matrix_l_;
  Matrix a_const_;
  Column eigen_values_;
};

}  // namespace linear
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_LINEAR_DOUBLE_HPP_
