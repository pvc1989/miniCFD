//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_POLYNOMIAL_PROJECTION_HPP_
#define MINI_POLYNOMIAL_PROJECTION_HPP_

#include <cmath>
#include <iostream>
#include <type_traits>
#include <utility>

#include "mini/algebra/eigen.hpp"
#include "mini/integrator/function.hpp"
#include "mini/polynomial/basis.hpp"

namespace mini {
namespace polynomial {

/**
 * @brief A vector-valued function projected onto an given orthonormal basis.
 * 
 * @tparam Scalar the data type of scalar components
 * @tparam kDim the dimension of the underlying physical space
 * @tparam kDegree the degree of completeness
 * @tparam kFunc the number of function components
 */
template <typename Scalar, int kDim, int kDegree, int kFunc>
class Projection {
 public:
  using Basis = OrthoNormal<Scalar, kDim, kDegree>;
  static constexpr int N = Basis::N;
  static constexpr int K = kFunc;
  using Coord = typename Basis::Coord;
  using MatNx1 = typename Basis::MatNx1;
  using MatNxN = typename Basis::MatNxN;
  using Mat1xN = algebra::Matrix<Scalar, 1, N>;
  using MatKxN = algebra::Matrix<Scalar, K, N>;
  using MatKx1 = algebra::Matrix<Scalar, K, 1>;
  using MatKxK = algebra::Matrix<Scalar, K, K>;
  using Value = MatKx1;
  using Coeff = MatKxN;

 public:
  template <typename Callable>
  Projection(Callable&& func, const Basis& basis)
      : basis_ptr_(&basis) {
    using Return = decltype(func(basis.center()));
    static_assert(std::is_same_v<Return, MatKx1> || std::is_scalar_v<Return>);
    coeff_ = integrator::Integrate([&](Coord const& xyz) {
      auto f_col = func(xyz);
      Mat1xN b_row = basis(xyz).transpose();
      MatKxN prod = f_col * b_row;
      return prod;
    }, basis.GetGauss());
    Coeff temp = coeff_ * basis.coeff();
    coeff_ = temp;
  }
  explicit Projection(const Basis& basis)
      : basis_ptr_(&basis) {
    coeff_.setZero();
  }
  Projection()
      : basis_ptr_(nullptr) {
    coeff_.setZero();
  }
  Projection(const Projection&) = default;
  Projection(Projection&&) noexcept = default;
  Projection& operator=(const Projection&) = default;
  Projection& operator=(Projection&&) noexcept = default;
  ~Projection() noexcept = default;

  MatKx1 operator()(Coord const& global) const {
    Coord local = global; local -= center();
    MatNx1 col = Raw<Scalar, kDim, kDegree>::CallAt(local);
    return coeff_ * col;
  }
  MatKxN GetCoeffOnOrthoNormalBasis() const {
    const auto& mat_a = basis_ptr_->coeff();
    MatKxN mat_x = coeff_;
    for (int i = N-1; i >= 0; --i) {
      for (int j = i+1; j < N; ++j) {
        mat_x.col(i) -= mat_x.col(j) * mat_a(j, i);
      }
      mat_x.col(i) /= mat_a(i, i);
    }
    return mat_x;
  }
  const MatKxN& GetCoeffOnRawBasis() const {
    return coeff_;
  }
  const Coord& center() const {
    return basis_ptr_->center();
  }
  const MatKxN& coeff() const {
    return coeff_;
  }
  MatKxN& coeff() {
    return coeff_;
  }
  MatKxN GetPdvValue(Coord const& global) const {
    auto local = global; local -= center();
    return Raw<Scalar, kDim, kDegree>::GetPdvValue(local, coeff());
  }
  MatKx1 GetAverage() const {
    const auto& mat_a = basis_ptr_->coeff();
    MatKxN mat_x = GetCoeffOnOrthoNormalBasis();
    mat_x.col(0) *= mat_a(0, 0);
    return mat_x.col(0);
  }
  MatKx1 GetSmoothness() const {
    auto mat_pdv_func = [&](Coord const& xyz) {
      auto mat_pdv = GetPdvValue(xyz);
      mat_pdv = mat_pdv.cwiseProduct(mat_pdv);
      return mat_pdv;
    };
    auto integral = integrator::Integrate(mat_pdv_func, basis_ptr_->GetGauss());
    auto volume = basis_ptr_->Measure();
    return Raw<Scalar, kDim, kDegree>::GetSmoothness(integral, volume);
  }
  template <typename Callable>
  void Project(Callable&& func, const Basis& basis) {
    *this = Projection(std::forward<Callable>(func), basis);
  }
  Projection& LeftMultiply(const MatKxK& left) {
    Coeff temp = left * coeff_;
    coeff_ = temp;
    return *this;
  }
  Projection& operator*=(const Scalar& ratio) {
    coeff_ *= ratio;
    return *this;
  }
  Projection& operator/=(const Scalar& ratio) {
    coeff_ /= ratio;
    return *this;
  }
  Projection& operator*=(const MatKx1& ratio) {
    for (int i = 0; i < K; ++i) {
      coeff_.row(i) *= ratio[i];
    }
    return *this;
  }
  Projection& operator+=(const MatKx1& offset) {
    coeff_.col(0) += offset;
    return *this;
  }
  Projection& operator+=(const Projection& that) {
    assert(this->basis_ptr_ == that.basis_ptr_);
    coeff_ += that.coeff_;
    return *this;
  }
  template <class T>
  void UpdateCoeffs(const T* new_coeffs) {
    for (int c = 0; c < N; ++c) {
      for (int r = 0; r < K; ++r) {
        coeff_(r, c) = *new_coeffs++;
      }
    }
  }
  void UpdateCoeffs(const MatKxN &new_coeff) {
    coeff_ = new_coeff;
  }

 public:
  MatKxN coeff_;
  const Basis* basis_ptr_;
};

}  // namespace polynomial
}  // namespace mini

#endif  // MINI_POLYNOMIAL_PROJECTION_HPP_
