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
  static constexpr int P = kDimensions;
  using Gauss = typename Basis::Gauss;
  using Local = typename Gauss::Local;
  using Global = typename Gauss::Global;
  using MatNx1 = typename Basis::MatNx1;
  using MatNxN = typename Basis::MatNxN;
  using Mat1xN = algebra::Matrix<Scalar, 1, N>;
  using MatKxK = algebra::Matrix<Scalar, K, K>;
  using Coeff = algebra::Matrix<Scalar, K, N>;
  using Value = algebra::Matrix<Scalar, K, 1>;

 private:
  Coeff coeff_;
  Basis basis_;

 public:
  explicit Projection(const Gauss &gauss)
      : basis_(gauss) {
  }
  Projection() = default;
  Projection(const Projection &) = default;
  Projection &operator=(const Projection &) = default;
  Projection &operator=(Projection &&that) noexcept {
    coeff_ = std::move(that.coeff_);
    basis_ = std::move(that.basis_);
    return *this;
  }
  Projection(Projection &&that) noexcept {
    *this = std::move(that);
  }
  ~Projection() noexcept = default;

  Value GlobalToValue(Global const &global) const {
    Local local = global; local -= center();
    MatNx1 col = basis::Taylor<Scalar, kDimensions, kDegrees>::GetValue(local);
    return coeff_ * col;
  }
  Value operator()(Global const &global) const {
    return GlobalToValue(global);
  }
  Coeff GetCoeffOnOrthoNormalBasis() const {
    auto const &mat_a = basis_.coeff();
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
  Basis const &basis() const {
    return basis_;
  }
  Gauss const &gauss() const {
    return basis().GetGauss();
  }
  Global const &center() const {
    return basis_.center();
  }
  Coeff const &coeff() const {
    return coeff_;
  }
  Coeff &coeff() {
    return coeff_;
  }
  Coeff GetPdvValue(Global const &global) const {
    auto local = global; local -= center();
    return basis::Taylor<Scalar, kDimensions, kDegrees>::GetPdvValue(local, coeff());
  }
  Value GetAverage() const {
    auto const &mat_a = basis_.coeff();
    Coeff mat_x = GetCoeffOnOrthoNormalBasis();
    mat_x.col(0) *= mat_a(0, 0);
    return mat_x.col(0);
  }
  Value GetSmoothness() const {
    auto mat_pdv_func = [&](Global const &xyz) {
      auto mat_pdv = GetPdvValue(xyz);
      mat_pdv = mat_pdv.cwiseProduct(mat_pdv);
      return mat_pdv;
    };
    auto integral = gauss::Integrate(mat_pdv_func, gauss());
    auto volume = basis_.Measure();
    return basis::Taylor<Scalar, kDimensions, kDegrees>::GetSmoothness(
        integral, volume);
  }
  template <typename Callable>
  void Approximate(Callable &&func) {
    using Return = decltype(func(basis_.center()));
    static_assert(std::is_same_v<Return, Value> || std::is_scalar_v<Return>);
    coeff_ = gauss::Integrate([&](Global const &xyz) {
      auto f_col = func(xyz);
      Mat1xN b_row = basis_(xyz).transpose();
      Coeff prod = f_col * b_row;
      return prod;
    }, gauss());
    Coeff temp = coeff_ * basis_.coeff();
    coeff_ = temp;
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
    assert(this->center() == that.center());
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
