//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_INTEGRATOR_PROJECTION_HPP_
#define MINI_INTEGRATOR_PROJECTION_HPP_

#include <cmath>
#include <iostream>
#include <type_traits>

#include "mini/algebra/eigen.hpp"
#include "mini/integrator/basis.hpp"
#include "mini/integrator/function.hpp"

namespace mini {
namespace integrator {

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

 public:
  template <typename Callable>
  Projection(Callable&& func, const Basis& basis)
      : basis_ptr_(&basis) {
    using Return = decltype(func(basis.GetCenter()));
    static_assert(std::is_same_v<Return, MatKx1> || std::is_scalar_v<Return>);
    coef_ = Integrate([&](Coord const& xyz) {
      auto f_col = func(xyz);
      Mat1xN b_row = basis(xyz).transpose();
      MatKxN prod = f_col * b_row;
      return prod;
    }, basis.GetGauss());
    average_ = coef_.col(0);
    average_ *= basis_ptr_->GetCoef()(0, 0);
    coef_ = coef_ * basis.GetCoef();
  }
  Projection()
      : coef_(MatKxN::Zero()), average_(MatKx1::Zero()), basis_ptr_(nullptr) {
  }
  Projection(const Projection&) = default;
  Projection(Projection&&) noexcept = default;
  Projection& operator=(const Projection&) = default;
  Projection& operator=(Projection&&) noexcept = default;
  ~Projection() noexcept = default;

  MatKx1 operator()(Coord const& xyz) const {
    MatNx1 col = RawBasis<Scalar, kDim, kOrder>::CallAt(xyz);
    return coef_ * col;
  }
  const MatKxN& GetCoef() const {
    return coef_;
  }
  MatKxN GetPdvValue(Coord const& global) const {
    MatKxN res; res.setZero();
    auto local = global; local -= basis_ptr_->GetCenter();
    return RawBasis<Scalar, kDim, kOrder>::GetPdvValue(local, GetCoef());
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
    auto integral = Integrate(mat_pdv_prod, basis_ptr_->GetGauss());
    auto volume = Integrate(
        [](const Coord &){ return 1.0; }, basis_ptr_->GetGauss());
    return RawBasis<Scalar, kDim, kOrder>::GetSmoothness(integral, volume);
  }
  template <typename Callable>
  void Project(Callable&& func, const Basis& basis) {
    *this = Projection(std::forward<Callable>(func), basis);
  }
  Projection& operator*=(const Scalar& ratio) {
    coef_ *= ratio;
    average_ *= ratio;
    return *this;
  }
  Projection& operator*=(const MatKx1& ratio) {
    for (int i = 0; i < K; ++i) {
      coef_.row(i) *= ratio[i];
      average_[i] *= ratio[i];
    }
    return *this;
  }
  Projection& operator+=(const MatKx1& offset) {
    coef_.col(0) += offset;
    average_ += offset;
    return *this;
  }
  Projection& operator+=(const Projection& that) {
    assert(this->basis_ptr_ == that.basis_ptr_);
    coef_ += that.coef_;
    average_ += that.average_;
    return *this;
  }

 private:
  MatKxN coef_;
  MatKx1 average_;
  const Basis* basis_ptr_;
};

}  // namespace integrator
}  // namespace mini

#endif  // MINI_INTEGRATOR_PROJECTION_HPP_
