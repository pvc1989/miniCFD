//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_POLYNOMIAL_BASIS_HPP_
#define MINI_POLYNOMIAL_BASIS_HPP_

#include <concepts>

#include <cassert>
#include <cmath>

#include <iostream>
#include <type_traits>

#include "mini/algebra/eigen.hpp"

#include "mini/gauss/function.hpp"
#include "mini/gauss/face.hpp"
#include "mini/gauss/cell.hpp"

#include "mini/polynomial/taylor.hpp"

namespace mini {
namespace polynomial {

/**
 * @brief A basis of the linear space formed by polynomials less than or equal to a given degree.
 * 
 * @tparam Scalar the data type of scalar components
 * @tparam kDimensions the dimension of the underlying physical space
 * @tparam kDegrees the degree of completeness
 */
template <std::floating_point Scalar, int kDimensions, int kDegrees>
class Linear {
  using TaylorBasis = Taylor<Scalar, kDimensions, kDegrees>;

 public:
  static constexpr int N = TaylorBasis::N;
  using Coord = typename TaylorBasis::Coord;
  using MatNx1 = typename TaylorBasis::MatNx1;
  using MatNxN = algebra::Matrix<Scalar, N, N>;
  using Gauss = std::conditional_t<kDimensions == 2,
      gauss::Face<Scalar, 2>, gauss::Cell<Scalar>>;

 public:
  explicit Linear(Coord const &center)
      : center_(center) {
    coeff_.setIdentity();
  }
  Linear() {
    center_.setZero();
    coeff_.setIdentity();
  }
  Linear(const Linear &) = default;
  Linear(Linear &&) noexcept = default;
  Linear &operator=(const Linear &) = default;
  Linear &operator=(Linear &&) noexcept = default;
  ~Linear() noexcept = default;

  MatNx1 operator()(Coord const &point) const {
    MatNx1 col = TaylorBasis::GetValue(point - center_);
    MatNx1 res = algebra::GetLowerTriangularView(coeff_) * col;
    return res;
  }
  Coord const &center() const {
    return center_;
  }
  MatNxN const &coeff() const {
    return coeff_;
  }
  void Transform(MatNxN const &a) {
    MatNxN temp = a * coeff_;
    coeff_ = temp;
  }
  void Transform(algebra::LowerTriangularView<MatNxN> const &a) {
    MatNxN temp;
    algebra::GetLowerTriangularView(&temp) = a * coeff_;
    algebra::GetLowerTriangularView(&coeff_) = temp;
  }
  void Shift(const Coord &new_center) {
    center_ = new_center;
  }

 private:
  Coord center_;
  MatNxN coeff_;
};

template <std::floating_point Scalar, int kDimensions, int kDegrees>
class OrthoNormal {
  using TaylorBasis = Taylor<Scalar, kDimensions, kDegrees>;
  using LinearBasis = Linear<Scalar, kDimensions, kDegrees>;

 public:
  static constexpr int N = LinearBasis::N;
  using Coord = typename LinearBasis::Coord;
  using Gauss = typename LinearBasis::Gauss;
  using MatNx1 = typename LinearBasis::MatNx1;
  using MatNxN = typename LinearBasis::MatNxN;
  using MatNxD = algebra::Matrix<Scalar, N, kDimensions>;

 public:
  explicit OrthoNormal(const Gauss &gauss)
      : gauss_ptr_(&gauss), basis_(gauss.center()) {
    assert(gauss.PhysDim() == kDimensions);
    OrthoNormalize(&basis_, gauss);
  }
  OrthoNormal() = default;
  OrthoNormal(const OrthoNormal &) = default;
  OrthoNormal(OrthoNormal &&) noexcept = default;
  OrthoNormal &operator=(const OrthoNormal &) = default;
  OrthoNormal &operator=(OrthoNormal &&) noexcept = default;
  ~OrthoNormal() noexcept = default;

  Coord const &center() const {
    return basis_.center();
  }
  MatNxN const &coeff() const {
    return basis_.coeff();
  }
  Gauss const &GetGauss() const {
    return *gauss_ptr_;
  }
  MatNx1 operator()(const Coord &global) const {
    auto local = global;
    local -= center();
    MatNx1 col = TaylorBasis::GetValue(local);
    return coeff() * col;
  }
  Scalar Measure() const {
    auto v = basis_.coeff()(0, 0);
    return 1 / (v * v);
  }
  MatNxD GetGradValue(const Coord &global) const {
    auto local = global;
    local -= center();
    return TaylorBasis::GetGradValue(local, coeff());
  }

 private:
  Gauss const *gauss_ptr_;
  Linear<Scalar, kDimensions, kDegrees> basis_;
};

}  // namespace polynomial
}  // namespace mini

#endif  // MINI_POLYNOMIAL_BASIS_HPP_
