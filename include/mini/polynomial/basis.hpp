//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_POLYNOMIAL_BASIS_HPP_
#define MINI_POLYNOMIAL_BASIS_HPP_

#include <cmath>
#include <iostream>
#include <type_traits>

#include "mini/algebra/eigen.hpp"

#include "mini/integrator/function.hpp"
#include "mini/integrator/face.hpp"
#include "mini/integrator/cell.hpp"

namespace mini {
namespace polynomial {

template <typename Scalar, int kDim, int kOrder>
class RawBasis;

template <typename Scalar>
class RawBasis<Scalar, 2, 2> {
 public:
  static constexpr int N = 6;  // the number of components
  using MatNx1 = algebra::Matrix<Scalar, N, 1>;
  using Coord = algebra::Matrix<Scalar, 2, 1>;

  static MatNx1 CallAt(const Coord &xy) {
    auto x = xy[0], y = xy[1];
    MatNx1 col = { 1, x, y, x * x, x * y, y * y };
    return col;
  }
};

template <typename Scalar>
class RawBasis<Scalar, 3, 2> {
 public:
  static constexpr int N = 10;  // the number of components
  using MatNx1 = algebra::Matrix<Scalar, N, 1>;
  using Coord = algebra::Matrix<Scalar, 3, 1>;

  // TODO(PVC): CallAt -> GetValue
  static MatNx1 CallAt(const Coord &xyz) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    MatNx1 col = { 1, x, y, z, x * x, x * y, x * z, y * y, y * z, z * z };
    return col;
  }
  template <typename MatKxN>
  static MatKxN GetPdvValue(const Coord &xyz, const MatKxN &coeff) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    MatKxN res; res.setZero();
    // pdv_x
    res.col(1) += coeff.col(1);
    res.col(1) += coeff.col(4) * (2 * x);
    res.col(1) += coeff.col(5) * y;
    res.col(1) += coeff.col(6) * z;
    // pdv_y
    res.col(2) += coeff.col(2);
    res.col(2) += coeff.col(5) * x;
    res.col(2) += coeff.col(7) * (2 * y);
    res.col(2) += coeff.col(8) * z;
    // pdv_z
    res.col(3) += coeff.col(3);
    res.col(3) += coeff.col(6) * x;
    res.col(3) += coeff.col(8) * y;
    res.col(3) += coeff.col(9) * (2 * z);
    // pdv_xx
    res.col(4) += coeff.col(4);
    // pdv_xy
    res.col(5) += coeff.col(5);
    // pdv_xz
    res.col(6) += coeff.col(6);
    // pdv_yy
    res.col(7) += coeff.col(7);
    // pdv_yz
    res.col(8) += coeff.col(8);
    // pdv_zz
    res.col(9) += coeff.col(9);
    return res;
  }
  template <int K>
  static auto GetSmoothness(
      const algebra::Matrix<Scalar, K, N> &integral, const Scalar &volume) {
    using MatKx1 = algebra::Matrix<Scalar, K, 1>;
    MatKx1 smoothness = integral.col(1);
    smoothness += integral.col(2);
    smoothness += integral.col(3);
    auto w2 = volume / 4;  // weight of 2nd-order partial derivatives
    smoothness += integral.col(4) * w2;
    smoothness += integral.col(5) * w2;
    smoothness += integral.col(6) * w2;
    smoothness += integral.col(7) * w2;
    smoothness += integral.col(8) * w2;
    smoothness += integral.col(9) * w2;
    return smoothness;
  }
};

/**
 * @brief A basis of the linear space formed by polynomials less than or equal to a given degree.
 * 
 * @tparam Scalar the data type of scalar components
 * @tparam kDim the dimension of the underlying physical space
 * @tparam kOrder the degree of completeness
 */
template <typename Scalar, int kDim, int kOrder>
class Basis {
  using Raw = RawBasis<Scalar, kDim, kOrder>;

 public:
  static constexpr int N = Raw::N;
  using Coord = typename Raw::Coord;
  using MatNx1 = typename Raw::MatNx1;
  using MatNxN = algebra::Matrix<Scalar, N, N>;
  using Gauss = std::conditional_t<kDim == 2, integrator::Face<Scalar, 2>, integrator::Cell<Scalar>>;

 public:
  explicit Basis(Coord const& center)
      : center_(center) {
  }
  Basis() {
    integrator::SetZero(&center_);
  }
  Basis(const Basis&) = default;
  Basis(Basis&&) noexcept = default;
  Basis& operator=(const Basis&) = default;
  Basis& operator=(Basis&&) noexcept = default;
  ~Basis() noexcept = default;

  MatNx1 operator()(Coord const& point) const {
    MatNx1 col = Raw::CallAt(point - center_);
    return coeff_ * col;
  }
  Coord const& GetCenter() const {
    return center_;
  }
  MatNxN const& GetCoeff() const {
    return coeff_;
  }
  void Transform(MatNxN const& a) {
    coeff_ = a * coeff_;
  }
  void Shift(const Coord& new_center) {
    center_ = new_center;
  }

 private:
  Coord center_;
  MatNxN coeff_ = MatNxN::Identity();
};

template <typename Scalar, int kDim, int kOrder>
class OrthoNormalBasis {
  using RB = RawBasis<Scalar, kDim, kOrder>;
  using GB = Basis<Scalar, kDim, kOrder>;

 public:
  static constexpr int N = GB::N;
  using Coord = typename GB::Coord;
  using Gauss = typename GB::Gauss;
  using MatNx1 = typename GB::MatNx1;
  using MatNxN = typename GB::MatNxN;

 public:
  explicit OrthoNormalBasis(const Gauss& gauss)
      : gauss_ptr_(&gauss), basis_(gauss.GetCenter()) {
    assert(gauss.PhysDim() == kDim);
    OrthoNormalize(&basis_, gauss);
  }
  OrthoNormalBasis() = default;
  OrthoNormalBasis(const OrthoNormalBasis&) = default;
  OrthoNormalBasis(OrthoNormalBasis&&) noexcept = default;
  OrthoNormalBasis& operator=(const OrthoNormalBasis&) = default;
  OrthoNormalBasis& operator=(OrthoNormalBasis&&) noexcept = default;
  ~OrthoNormalBasis() noexcept = default;

  Coord const& GetCenter() const {
    return basis_.GetCenter();
  }
  MatNxN const& GetCoeff() const {
    return basis_.GetCoeff();
  }
  Gauss const& GetGauss() const {
    return *gauss_ptr_;
  }
  MatNx1 operator()(const Coord& global) const {
    auto local = global;
    local -= GetCenter();
    MatNx1 col = RB::CallAt(local);
    return GetCoeff() * col;
  }
  Scalar Measure() const {
    auto v = basis_.GetCoeff()(0, 0);
    return 1 / (v * v);
  }

 private:
  Gauss const* gauss_ptr_;
  Basis<Scalar, kDim, kOrder> basis_;
};

}  // namespace polynomial
}  // namespace mini

#endif  // MINI_POLYNOMIAL_BASIS_HPP_
