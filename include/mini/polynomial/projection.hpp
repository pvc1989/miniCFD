//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_POLYNOMIAL_PROJECTION_HPP_
#define MINI_POLYNOMIAL_PROJECTION_HPP_

#include <concepts>

#include <cmath>

#include <iostream>
#include <type_traits>
#include <utility>

#include "mini/algebra/eigen.hpp"
#include "mini/gauss/function.hpp"
#include "mini/basis/linear.hpp"

namespace mini {
namespace polynomial {

/**
 * @brief A vector-valued function projected onto an given orthonormal basis.
 * 
 * @tparam Scalar the data type of scalar components
 * @tparam kDimensions the dimension of the underlying physical space
 * @tparam kDegrees the degree of completeness
 * @tparam kComponents the number of function components
 */
template <std::floating_point Scalar, int kDimensions, int kDegrees,
    int kComponents>
class Projection {
 public:
  using Basis = basis::OrthoNormal<Scalar, kDimensions, kDegrees>;
  static constexpr int N = Basis::N;
  static constexpr int K = kComponents;
  using Coord = typename Basis::Coord;
  using MatNx1 = typename Basis::MatNx1;
  using MatNxN = typename Basis::MatNxN;
  using Mat1xN = algebra::Matrix<Scalar, 1, N>;
  using MatKxK = algebra::Matrix<Scalar, K, K>;
  using Coeff = algebra::Matrix<Scalar, K, N>;
  using Value = algebra::Matrix<Scalar, K, 1>;

 private:
  Coeff coeff_;
  const Basis *basis_ptr_;

 public:
  template <typename Callable>
  Projection(Callable &&func, const Basis &basis)
      : basis_ptr_(&basis) {
    using Return = decltype(func(basis.center()));
    static_assert(std::is_same_v<Return, Value> || std::is_scalar_v<Return>);
    coeff_ = gauss::Integrate([&](Coord const &xyz) {
      auto f_col = func(xyz);
      Mat1xN b_row = basis(xyz).transpose();
      Coeff prod = f_col * b_row;
      return prod;
    }, basis.GetGauss());
    Coeff temp = coeff_ * basis.coeff();
    coeff_ = temp;
  }
  explicit Projection(const Basis &basis)
      : basis_ptr_(&basis) {
    coeff_.setZero();
  }
  Projection()
      : basis_ptr_(nullptr) {
    coeff_.setZero();
  }
  Projection(const Projection &) = default;
  Projection &operator=(const Projection &) = default;
  Projection &operator=(Projection &&that) noexcept {
    coeff_ = std::move(that.coeff_);
    basis_ptr_ = that.basis_ptr_;
    return *this;
  }
  Projection(Projection &&that) noexcept {
    *this = std::move(that);
  }
  ~Projection() noexcept = default;

  Value operator()(Coord const &global) const {
    Coord local = global; local -= center();
    MatNx1 col = basis::Taylor<Scalar, kDimensions, kDegrees>::GetValue(local);
    return coeff_ * col;
  }
  Coeff GetCoeffOnOrthoNormalBasis() const {
    auto const &mat_a = basis_ptr_->coeff();
    Coeff mat_x = coeff_;
    for (int i = N-1; i >= 0; --i) {
      for (int j = i+1; j < N; ++j) {
        mat_x.col(i) -= mat_x.col(j) * mat_a(j, i);
      }
      mat_x.col(i) /= mat_a(i, i);
    }
    return mat_x;
  }
  Coeff const &GetCoeffOnTaylorBasis() const {
    return coeff_;
  }
  Coord const &center() const {
    return basis_ptr_->center();
  }
  Coeff const &coeff() const {
    return coeff_;
  }
  Coeff &coeff() {
    return coeff_;
  }
  Coeff GetPdvValue(Coord const &global) const {
    auto local = global; local -= center();
    return basis::Taylor<Scalar, kDimensions, kDegrees>::GetPdvValue(local, coeff());
  }
  Value GetAverage() const {
    auto const &mat_a = basis_ptr_->coeff();
    Coeff mat_x = GetCoeffOnOrthoNormalBasis();
    mat_x.col(0) *= mat_a(0, 0);
    return mat_x.col(0);
  }
  Value GetSmoothness() const {
    auto mat_pdv_func = [&](Coord const &xyz) {
      auto mat_pdv = GetPdvValue(xyz);
      mat_pdv = mat_pdv.cwiseProduct(mat_pdv);
      return mat_pdv;
    };
    auto integral = gauss::Integrate(mat_pdv_func, basis_ptr_->GetGauss());
    auto volume = basis_ptr_->Measure();
    return basis::Taylor<Scalar, kDimensions, kDegrees>::GetSmoothness(
        integral, volume);
  }
  template <typename Callable>
  void Project(Callable &&func, const Basis &basis) {
    *this = Projection(std::forward<Callable>(func), basis);
  }
  Projection &LeftMultiply(const MatKxK &left) {
    Coeff temp = left * coeff_;
    coeff_ = temp;
    return *this;
  }
  Projection &operator*=(Scalar ratio) {
    coeff_ *= ratio;
    return *this;
  }
  Projection &operator/=(Scalar ratio) {
    coeff_ /= ratio;
    return *this;
  }
  Projection &operator*=(const Value& ratio) {
    for (int i = 0; i < K; ++i) {
      coeff_.row(i) *= ratio[i];
    }
    return *this;
  }
  Projection &operator+=(const Value& offset) {
    coeff_.col(0) += offset;
    return *this;
  }
  Projection &operator+=(const Projection &that) {
    assert(this->basis_ptr_ == that.basis_ptr_);
    coeff_ += that.coeff_;
    return *this;
  }
  template <class T>
  void UpdateCoeffs(const T *new_coeffs) {
    for (int c = 0; c < N; ++c) {
      for (int r = 0; r < K; ++r) {
        coeff_(r, c) = *new_coeffs++;
      }
    }
  }
};

}  // namespace polynomial
}  // namespace mini

#endif  // MINI_POLYNOMIAL_PROJECTION_HPP_
