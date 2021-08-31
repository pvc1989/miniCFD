//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_INTEGRATOR_BASIS_HPP_
#define MINI_INTEGRATOR_BASIS_HPP_

#include <cmath>
#include <iostream>
#include <type_traits>

#include "mini/algebra/eigen.hpp"

#include "mini/integrator/function.hpp"
#include "mini/integrator/face.hpp"
#include "mini/integrator/cell.hpp"

namespace mini {
namespace integrator {

/**
 * @brief A class template to mimic vector-valued functions.
 * 
 * @tparam Scalar the type of scalar components
 * @tparam kDim the dimension of the underlying space
 * @tparam kOrder the degree of completeness
 */
template <typename Scalar, int kDim, int kOrder>
class Basis;

template <typename Scalar>
class Basis<Scalar, 2, 2> {
 public:
  static constexpr int N = 6;  // the number of components
  using Coord = algebra::Matrix<Scalar, 2, 1>;
  using MatNx1 = algebra::Matrix<Scalar, N, 1>;
  using MatNxN = algebra::Matrix<Scalar, N, N>;
  explicit Basis(Coord const& c = {0, 0})
      : center_(c) {
  }
  Basis(const Basis&) = default;
  Basis(Basis&&) noexcept = default;
  Basis& operator=(const Basis&) = default;
  Basis& operator=(Basis&&) noexcept = default;
  ~Basis() noexcept = default;

  MatNx1 operator()(Coord const& xy) const {
    auto x = xy[0] - center_[0], y = xy[1] - center_[1];
    MatNx1 col = { 1, x, y, x * x, x * y, y * y };
    return coef_ * col;
  }
  Coord const& GetCenter() const {
    return center_;
  }
  MatNxN const& GetCoef() const {
    return coef_;
  }
  void Transform(MatNxN const& a) {
    coef_ = a * coef_;
  }
  void Orthonormalize(const Face<Scalar, 2>& elem) {
    assert(elem.PhysDim() == 2);
    integrator::Orthonormalize(this, elem);
  }

 private:
  Coord center_;
  MatNxN coef_ = MatNxN::Identity();
};

template <typename Scalar>
class Basis<Scalar, 3, 2> {
 public:
  static constexpr int N = 10;  // the number of components
  using Coord = algebra::Matrix<Scalar, 3, 1>;
  using MatNx1 = algebra::Matrix<Scalar, N, 1>;
  using MatNxN = algebra::Matrix<Scalar, N, N>;
  explicit Basis(Coord const& c = {0, 0, 0})
      : center_(c) {
  }
  Basis(const Basis&) = default;
  Basis(Basis&&) noexcept = default;
  Basis& operator=(const Basis&) = default;
  Basis& operator=(Basis&&) noexcept = default;
  ~Basis() noexcept = default;

  MatNx1 operator()(Coord const& xyz) const {
    auto x = xyz[0] - center_[0], y = xyz[1] - center_[1],
         z = xyz[2] - center_[2];
    MatNx1 col = { 1, x, y, z, x * x, x * y, x * z, y * y, y * z, z * z };
    return coef_ * col;
  }
  Coord const& GetCenter() const {
    return center_;
  }
  MatNxN const& GetCoef() const {
    return coef_;
  }
  void Transform(MatNxN const& a) {
    coef_ = a * coef_;
  }
  void Shift(Coord const& new_center) {
    center_ = new_center;
  }
  void Orthonormalize(const Cell<Scalar>& elem) {
    assert(elem.PhysDim() == 3);
    integrator::Orthonormalize(this, elem);
  }

 private:
  Coord center_;
  MatNxN coef_ = MatNxN::Identity();
};

/**
 * @brief A class template to represent a vector-valued function projected onto an orthonormal basis.
 * 
 * @tparam Scalar the type of scalar components
 * @tparam kDim the dimension of the underlying space
 * @tparam kOrder the degree of completeness
 * @tparam kFunc the number of function components
 */
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

  template <typename Callable, typename Element>
  void Reset(Callable&& func, BasisType const& basis, Element const& elem) {
    center_ = basis.GetCenter();
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

#endif  // MINI_INTEGRATOR_BASIS_HPP_
