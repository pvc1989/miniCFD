// Copyright 2019 PEI Weicheng and YANG Minghao

#ifndef MINI_RIEMANN_LINEAR_MULTIPLE_HPP_
#define MINI_RIEMANN_LINEAR_MULTIPLE_HPP_

#include <Eigen/Eigenvalues>

#include "mini/algebra/eigen.hpp"

namespace mini {
namespace riemann {
namespace linear {

template <typename S, int K, int D>
class Multiple {
 public:
  static constexpr int kDimensions = D;
  static constexpr int kComponents = K;
  using Scalar = S;
  using Vector = algebra::Vector<Scalar, kDimensions>;
  using Column = algebra::Vector<Scalar, kComponents>;
  using Matrix = algebra::Matrix<Scalar, kComponents, kComponents>;
  using Jacobi = Matrix;
  using Coefficient = algebra::Vector<Jacobi, kDimensions>;
  using Conservative = Column;
  using Flux = Column;

  // Constructor:
  Multiple() = default;
  explicit Multiple(const Matrix &a_const) : a_const_(a_const) { Decompose(); }
  // Get F on T Axia
  Flux GetFluxOnTimeAxis(const Conservative &left, const Conservative &right)
      const {
    Flux flux; flux.setZero();
    for (int k = 0; k < K; ++k) {
      const auto &state = (eigen_values_[k] > 0 ? left : right);
      auto temp = eigen_cols_.row(k).dot(state) * eigen_values_[k];
      flux += eigen_rows_.col(k) * temp;
    }
    return flux;
  }
  // Get F of U
  Flux GetFlux(const Conservative &state) const {
    return a_const_ * state;
  }

 private:
  void Decompose() {
    auto solver = Eigen::EigenSolver<Matrix>(a_const_);
    eigen_values_ = solver.eigenvalues().real();
    eigen_rows_ = solver.eigenvectors().real();
    eigen_cols_ = eigen_rows_.inverse();
  }

  Matrix eigen_rows_;
  Matrix eigen_cols_;
  Matrix a_const_;
  Column eigen_values_;

 public:
  const Matrix& A() const {
    return a_const_;
  }
  const Matrix& L() const {
    return eigen_rows_;
  }
  const Matrix& R() const {
    return eigen_cols_;
  }
};

}  // namespace linear
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_LINEAR_MULTIPLE_HPP_
