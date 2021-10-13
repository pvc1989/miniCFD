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

  static MatNx1 CallAt(const Coord &xyz) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    MatNx1 col = { 1, x, y, z, x * x, x * y, x * z, y * y, y * z, z * z };
    return col;
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
class Basis : protected RawBasis<Scalar, kDim, kOrder> {
  using Raw = RawBasis<Scalar, kDim, kOrder>;

 public:
  using Raw::N;
  using typename Raw::Coord;
  using typename Raw::MatNx1;
  using MatNxN = algebra::Matrix<Scalar, N, N>;
  using Gauss = std::conditional_t<kDim == 2, Face<Scalar, 2>, Cell<Scalar>>;

 public:
  explicit Basis(Coord const& center)
      : center_(center) {
  }
  Basis() {
    SetZero(&center_);
  }
  Basis(const Basis&) = default;
  Basis(Basis&&) noexcept = default;
  Basis& operator=(const Basis&) = default;
  Basis& operator=(Basis&&) noexcept = default;
  ~Basis() noexcept = default;

  MatNx1 operator()(Coord const& point) const {
    MatNx1 col = Raw::CallAt(point - center_);
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
  void Shift(const Coord& new_center) {
    center_ = new_center;
  }
  void OrthoNormalize(const Gauss& gauss) {
    assert(gauss.PhysDim() == kDim);
    integrator::OrthoNormalize(this, gauss);
  }

 private:
  Coord center_;
  MatNxN coef_ = MatNxN::Identity();
};

template <typename Scalar, int kDim, int kOrder>
class OrthoNormalBasis : protected Basis<Scalar, kDim, kOrder> {
  using RB = RawBasis<Scalar, kDim, kOrder>;
  using GB = Basis<Scalar, kDim, kOrder>;

 public:
  using GB::N;  // the number of components
  using typename GB::Coord;
  using typename GB::Gauss;
  using typename GB::MatNx1;
  using typename GB::MatNxN;

 public:
  explicit OrthoNormalBasis(const Gauss& gauss)
      : gauss_ptr_(&gauss), basis_(gauss.GetCenter()) {
    assert(gauss.PhysDim() == kDim);
    OrthoNormalize(&basis_, gauss);
  }
  OrthoNormalBasis(const OrthoNormalBasis&) = default;
  OrthoNormalBasis(OrthoNormalBasis&&) noexcept = default;
  OrthoNormalBasis& operator=(const OrthoNormalBasis&) = default;
  OrthoNormalBasis& operator=(OrthoNormalBasis&&) noexcept = default;
  ~OrthoNormalBasis() noexcept = default;

  Coord const& GetCenter() const {
    return basis_.GetCenter();
  }
  MatNxN const& GetCoef() const {
    return basis_.GetCoef();
  }
  Gauss const& GetGauss() const {
    return *gauss_ptr_;
  }
  MatNx1 operator()(const Coord& global) const {
    auto local = global;
    global -= GetCenter();
    MatNx1 col = RB::CallAt(local);
    return GetCoef() * col;
  }

 private:
  Gauss const* gauss_ptr_;
  Basis<Scalar, kDim, kOrder> basis_;
};

}  // namespace integrator
}  // namespace mini

#endif  // MINI_INTEGRATOR_BASIS_HPP_
