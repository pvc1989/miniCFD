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
 * @tparam kOrder the degree of completeness
 * @tparam kFunc the number of function components
 */
template <typename Scalar, int kDim, int kOrder, int kFunc>
class Projection {
 private:
  using Basis = OrthoNormalBasis<Scalar, kDim, kOrder>;

 public:
  static constexpr int N = Basis::N;
  static constexpr int K = kFunc;
  using Coord = typename Basis::Coord;
  using MatNx1 = typename Basis::MatNx1;
  using MatNxN = typename Basis::MatNxN;
  using Mat1xN = algebra::Matrix<Scalar, 1, N>;
  using MatKxN = algebra::Matrix<Scalar, K, N>;
  using MatKx1 = algebra::Matrix<Scalar, K, 1>;
  using MatKxK = algebra::Matrix<Scalar, K, K>;

 public:
  template <typename Callable>
  Projection(Callable&& func, const Basis& basis)
      : basis_ptr_(&basis) {
    using Return = decltype(func(basis.GetCenter()));
    static_assert(std::is_same_v<Return, MatKx1> || std::is_scalar_v<Return>);
    coeff_ = integrator::Integrate([&](Coord const& xyz) {
      auto f_col = func(xyz);
      Mat1xN b_row = basis(xyz).transpose();
      MatKxN prod = f_col * b_row;
      return prod;
    }, basis.GetGauss());
    average_ = coeff_.col(0);
    average_ *= basis_ptr_->GetCoeff()(0, 0);
    coeff_ = coeff_ * basis.GetCoeff();
  }
  explicit Projection(const Basis& basis)
      : coeff_(MatKxN::Zero()), average_(MatKx1::Zero()), basis_ptr_(&basis) {
  }
  Projection()
      : coeff_(MatKxN::Zero()), average_(MatKx1::Zero()), basis_ptr_(nullptr) {
  }
  Projection(const Projection&) = default;
  Projection(Projection&&) noexcept = default;
  Projection& operator=(const Projection&) = default;
  Projection& operator=(Projection&&) noexcept = default;
  ~Projection() noexcept = default;

  MatKx1 operator()(Coord const& global) const {
    auto local = global; local -= basis_ptr_->GetCenter();
    MatNx1 col = RawBasis<Scalar, kDim, kOrder>::CallAt(local);
    return coeff_ * col;
  }
  const MatKxN& GetCoeff() const {
    return coeff_;
  }
  MatKxN GetPdvValue(Coord const& global) const {
    MatKxN res; res.setZero();
    auto local = global; local -= basis_ptr_->GetCenter();
    return RawBasis<Scalar, kDim, kOrder>::GetPdvValue(local, GetCoeff());
  }
  MatKx1 const& GetAverage() const {
    return average_;
  }
  MatKx1 GetSmoothness() const {
    auto mat_pdv_prod = [&](Coord const& xyz) {
      auto mat_pdv = GetPdvValue(xyz);
      mat_pdv = mat_pdv.cwiseProduct(mat_pdv);
      return mat_pdv;
    };
    auto integral = integrator::Integrate(mat_pdv_prod, basis_ptr_->GetGauss());
    auto volume = basis_ptr_->Measure();
    return RawBasis<Scalar, kDim, kOrder>::GetSmoothness(integral, volume);
  }
  template <typename Callable>
  void Project(Callable&& func, const Basis& basis) {
    *this = Projection(std::forward<Callable>(func), basis);
  }
  Projection& LeftMultiply(const MatKxK& left) {
    coeff_ = left * coeff_;
    average_ = left * average_;
    return *this;
  }
  Projection& operator*=(const Scalar& ratio) {
    coeff_ *= ratio;
    average_ *= ratio;
    return *this;
  }
  Projection& operator/=(const Scalar& ratio) {
    coeff_ /= ratio;
    average_ /= ratio;
    return *this;
  }
  Projection& operator*=(const MatKx1& ratio) {
    for (int i = 0; i < K; ++i) {
      coeff_.row(i) *= ratio[i];
      average_[i] *= ratio[i];
    }
    return *this;
  }
  Projection& operator+=(const MatKx1& offset) {
    coeff_.col(0) += offset;
    average_ += offset;
    return *this;
  }
  Projection& operator+=(const Projection& that) {
    assert(this->basis_ptr_ == that.basis_ptr_);
    coeff_ += that.coeff_;
    average_ += that.average_;
    return *this;
  }
  void UpdateCoeffs(const Scalar* new_coeffs) {
    for (int c = 0; c < N; ++c) {
      for (int r = 0; r < K; ++r) {
        coeff_(r, c) = *new_coeffs++;
      }
    }
  }

 private:
  MatKxN coeff_;
  MatKx1 average_;
  const Basis* basis_ptr_;
};

}  // namespace polynomial
}  // namespace mini

#endif  // MINI_POLYNOMIAL_PROJECTION_HPP_
