//  Copyright 2023 PEI Weicheng
#ifndef MINI_POLYNOMIAL_HEXAHEDRON_HPP_
#define MINI_POLYNOMIAL_HEXAHEDRON_HPP_

#include <concepts>

#include <cmath>

#include <iostream>
#include <type_traits>
#include <utility>

#include "mini/algebra/eigen.hpp"
#include "mini/gauss/hexahedron.hpp"
#include "mini/basis/lagrange.hpp"

namespace mini {
namespace polynomial {

/**
 * @brief A vector-valued function interpolated on an given basis::lagrange::Hexahedron basis.
 * 
 * The interpolation nodes are collocated with quadrature points.
 * 
 * @tparam Scalar the data type of scalar components
 * @tparam kDegrees the degree of completeness
 * @tparam kComponents the number of function components
 */
template <std::floating_point Scalar, int kDegreeX, int kDegreeY, int kDegreeZ,
    int kComponents>
class Hexahedron {
 public:
  using Basis = basis::lagrange::Hexahedron< Scalar, kDegreeX, kDegreeY, kDegreeZ >;
  using Gauss = gauss::Hexahedron< Scalar, Basis::I, Basis::J, Basis::K >;
  using Lagrange = typename Gauss::Lagrange;
  static constexpr int N = Basis::N;
  static constexpr int K = kComponents;
  using Local = typename Gauss::Local;
  using Global = typename Gauss::Global;
  using MatKxN = algebra::Matrix<Scalar, K, N>;
  using MatKx1 = algebra::Matrix<Scalar, K, 1>;
  using Value = MatKx1;
  using Coeff = MatKxN;

 private:
  const Gauss *gauss_ptr_ = nullptr;
  const Basis *basis_ptr_ = nullptr;
  MatKxN coeff_;  // u^h(local) = coeff_ @ basis.GetValues(local)

 public:
  template <typename Callable>
  Hexahedron(Callable &&global_to_value, const Gauss &gauss, const Basis &basis)
      : gauss_ptr_(&gauss), basis_ptr_(&basis) {
    for (int ijk = 0; ijk < N; ++ijk) {
      auto &global = gauss_ptr_->GetGlobalCoord(ijk);
      coeff_.col(ijk) = global_to_value(global);  // value in physical space
      auto jacobian_det =
          gauss_ptr_->GetGlobalWeight(ijk) / gauss_ptr_->GetLocalWeight(ijk);
      coeff_.col(ijk) *= jacobian_det;  // value in parametric space
    }
  }
  explicit Hexahedron(const Gauss &gauss, const Basis &basis)
      : gauss_ptr_(&gauss), basis_ptr_(&basis) {
    coeff_.setZero();
  }
  Hexahedron() {
    coeff_.setZero();
  }
  Hexahedron(const Hexahedron &) = default;
  Hexahedron(Hexahedron &&) noexcept = default;
  Hexahedron &operator=(const Hexahedron &) = default;
  Hexahedron &operator=(Hexahedron &&) noexcept = default;
  ~Hexahedron() noexcept = default;

  Value LobalToValue(Local const &local) const {
    Value value = coeff_ * basis_ptr_->GetValues(local).transpose();
    value /= lagrange().LocalToJacobian(local).determinant();
    return value;
  }

  Value GlobalToValue(Global const &global) const {
    Local local = lagrange().GlobalToLocal(global);
    return LobalToValue(local);
  }

  Global const &center() const {
    return gauss_ptr_->center();
  }
  MatKxN const &coeff() const {
    return coeff_;
  }
  MatKxN &coeff() {
    return coeff_;
  }
  Gauss const &gauss() const {
    return *gauss_ptr_;
  }
  Lagrange const &lagrange() const {
    return gauss().lagrange();
  }
  template <typename Callable>
  void Interpolate(Callable &&func) {
    *this = Hexahedron(std::forward<Callable>(func), gauss_ptr_, basis_ptr_);
  }

  static Basis BuildInterpolationBasis() {
    auto line_x = typename Basis::LineX{ Gauss::GaussX::points };
    auto line_y = typename Basis::LineY{ Gauss::GaussY::points };
    auto line_z = typename Basis::LineZ{ Gauss::GaussZ::points };
    return Basis(line_x, line_y, line_z);
  }
};

}  // namespace polynomial
}  // namespace mini

#endif  // MINI_POLYNOMIAL_HEXAHEDRON_HPP_
