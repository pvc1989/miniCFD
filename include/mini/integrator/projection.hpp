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
    coef_ = coef_ * basis.GetCoef();
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

 private:
  MatKxN coef_;
  const Basis* basis_ptr_;
};

template <typename Scalar, int kDim, int kOrder, int kFunc>
class ProjFunc;

template <typename Scalar, int kFunc>
class ProjFunc<Scalar, 2, 2, kFunc> {
 public:
  using BasisType = Basis<Scalar, 2, 2>;
  using CoordType = typename BasisType::Coord;
  static constexpr int K = kFunc;
  static constexpr int N = BasisType::N;
  using Mat1xN = algebra::Matrix<Scalar, 1, N>;
  using MatNx1 = algebra::Matrix<Scalar, N, 1>;
  using MatNxN = algebra::Matrix<Scalar, N, N>;
  using MatKxN = algebra::Matrix<Scalar, K, N>;
  using MatKx1 = algebra::Matrix<Scalar, K, 1>;

  template <typename Callable, typename Element>
  ProjFunc(Callable&& func, BasisType const& basis, Element const& elem)
      : center_(basis.GetCenter()) {
    using Ret = decltype(func(center_));
    static_assert(std::is_same_v<Ret, MatKx1> || std::is_scalar_v<Ret>);
    coef_ = Integrate([&](CoordType const& xyz) {
      auto f_col = func(xyz);
      Mat1xN b_row = basis(xyz).transpose();
      MatKxN prod = f_col * b_row;
      return prod;
    }, elem);
    coef_ = coef_ * basis.GetCoef();
  }
  ProjFunc(const ProjFunc&) = default;
  ProjFunc(ProjFunc&&) noexcept = default;
  ProjFunc& operator=(const ProjFunc&) = default;
  ProjFunc& operator=(ProjFunc&&) noexcept = default;
  ~ProjFunc() noexcept = default;

  MatKx1 operator()(CoordType const& xy) const {
    auto x = xy[0] - center_[0], y = xy[1] - center_[1];
    MatNx1 col = { 1, x, y, x * x, x * y, y * y };
    return coef_ * col;
  }
  MatKxN GetCoef() const {
    return coef_;
  }

 private:
  CoordType center_;
  MatKxN coef_;
};

template <typename Scalar, int kFunc>
class ProjFunc<Scalar, 3, 2, kFunc> {
 public:
  using BasisType = Basis<Scalar, 3, 2>;
  using CoordType = typename BasisType::Coord;
  static constexpr int K = kFunc;
  static constexpr int N = BasisType::N;
  using Mat1xN = algebra::Matrix<Scalar, 1, N>;
  using MatNx1 = algebra::Matrix<Scalar, N, 1>;
  using MatNxN = algebra::Matrix<Scalar, N, N>;
  using MatKxN = algebra::Matrix<Scalar, K, N>;
  using MatKx1 = algebra::Matrix<Scalar, K, 1>;

  template <typename Callable, typename Element>
  ProjFunc(Callable&& func, BasisType const& basis, Element const& elem) {
    Reset(func, basis, elem);
  }
  ProjFunc() = default;
  ProjFunc(const ProjFunc&) = default;
  ProjFunc(ProjFunc&&) noexcept = default;
  ProjFunc& operator=(const ProjFunc&) = default;
  ProjFunc& operator=(ProjFunc&&) noexcept = default;
  ~ProjFunc() noexcept = default;

  MatKx1 operator()(CoordType const& xyz) const {
    auto x = xyz[0] - center_[0], y = xyz[1] - center_[1],
         z = xyz[2] - center_[2];
    MatNx1 col = { 1, x, y, z, x * x, x * y, x * z, y * y, y * z, z * z };
    return coef_ * col;
  }
  const MatKxN& GetCoef() const {
    return coef_;
  }

  MatKxN GetMpdv(CoordType const& xyz) const {
    auto x = xyz[0] - center_[0], y = xyz[1] - center_[1],
         z = xyz[2] - center_[2];
    MatKxN res;
    SetZero(&res);
    // pdv_x
    res.col(1) += coef_.col(1);
    res.col(1) += coef_.col(4) * (2 * x);
    res.col(1) += coef_.col(5) * y;
    res.col(1) += coef_.col(6) * z;
    // pdv_y
    res.col(2) += coef_.col(2);
    res.col(2) += coef_.col(5) * x;
    res.col(2) += coef_.col(7) * (2 * y);
    res.col(2) += coef_.col(8) * z;
    // pdv_z
    res.col(3) += coef_.col(3);
    res.col(3) += coef_.col(6) * x;
    res.col(3) += coef_.col(8) * y;
    res.col(3) += coef_.col(9) * (2 * z);
    // pdv_xx
    res.col(4) += coef_.col(4);
    // pdv_xy
    res.col(5) += coef_.col(5);
    // pdv_xz
    res.col(6) += coef_.col(6);
    // pdv_yy
    res.col(7) += coef_.col(7);
    // pdv_yz
    res.col(8) += coef_.col(8);
    // pdv_zz
    res.col(9) += coef_.col(9);
    return res;
  }

  template <typename Element>
  auto GetSmoothness(Element const& elem) {
    auto mat_pdv_prod = [&](CoordType const& xyz) {
      auto mat_pdv = GetMpdv(xyz);
      mat_pdv = mat_pdv.cwiseProduct(mat_pdv);
      return mat_pdv;
    };
    auto integral = Integrate(mat_pdv_prod, elem);
    MatKx1 smoothness = integral.col(1);
    smoothness += integral.col(2);
    smoothness += integral.col(3);
    auto volume = Integrate([](CoordType const& xyz){ return 1.0; }, elem);
    for (int i = 4; i < N; ++i) {
      smoothness += integral.col(i) * (volume / 4);
    }
    return smoothness;
  }

  template <typename Callable, typename Element>
  void Reset(Callable&& func, BasisType const& basis, Element const& elem) {
    center_ = basis.GetCenter();
    assert(center_ == elem.GetCenter());
    using Ret = decltype(func(center_));
    static_assert(std::is_same_v<Ret, MatKx1> || std::is_scalar_v<Ret>);
    coef_ = Integrate([&](CoordType const& xyz) {
      auto f_col = func(xyz);
      Mat1xN b_row = basis(xyz).transpose();
      MatKxN prod = f_col * b_row;
      return prod;
    }, elem);
    coef_ = coef_ * basis.GetCoef();
  }

 private:
  CoordType center_;
  MatKxN coef_;
};

}  // namespace integrator
}  // namespace mini

#endif  // MINI_INTEGRATOR_PROJECTION_HPP_
