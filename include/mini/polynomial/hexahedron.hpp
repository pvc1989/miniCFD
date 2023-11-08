//  Copyright 2023 PEI Weicheng
#ifndef MINI_POLYNOMIAL_HEXAHEDRON_HPP_
#define MINI_POLYNOMIAL_HEXAHEDRON_HPP_

#include <concepts>

#include <cmath>

#include <iostream>
#include <type_traits>
#include <utility>

#include "mini/algebra/eigen.hpp"
#include "mini/gauss/cell.hpp"
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
 * @tparam kDegreeX the degree of completeness in the 1st dimension
 * @tparam kDegreeY the degree of completeness in the 2nd dimension
 * @tparam kDegreeZ the degree of completeness in the 3rd dimension
 * @tparam kComponents the number of function components
 * @tparam kLocal in local (parametric) space or not
 */
template <std::floating_point Scalar, int kDegreeX, int kDegreeY, int kDegreeZ,
    int kComponents, bool kLocal = false>
class Hexahedron {
 public:
  using Basis = basis::lagrange::Hexahedron< Scalar, kDegreeX, kDegreeY, kDegreeZ >;
  using Gauss = gauss::Hexahedron< Scalar, Basis::I, Basis::J, Basis::K >;
  using GaussBase = gauss::Cell<Scalar>;
  using Lagrange = typename Gauss::Lagrange;
  static constexpr int N = Basis::N;
  static constexpr int K = kComponents;
  static constexpr int P = std::max({kDegreeX, kDegreeY, kDegreeZ});
  using Local = typename Gauss::Local;
  using Global = typename Gauss::Global;
  using Coeff = algebra::Matrix<Scalar, K, N>;
  using Value = algebra::Matrix<Scalar, K, 1>;

 private:
  static const Basis basis_;
  const Gauss *gauss_ptr_ = nullptr;
  Coeff coeff_;  // u^h(local) = coeff_ @ basis.GetValues(local)

 public:
  explicit Hexahedron(const GaussBase &gauss)
      : gauss_ptr_(dynamic_cast<const Gauss *>(&gauss)) {
  }
  Hexahedron() = default;
  Hexahedron(const Hexahedron &) = default;
  Hexahedron(Hexahedron &&) noexcept = default;
  Hexahedron &operator=(const Hexahedron &) = default;
  Hexahedron &operator=(Hexahedron &&) noexcept = default;
  ~Hexahedron() noexcept = default;

  Value LobalToValue(Local const &local) const {
    Value value = coeff_ * basis_.GetValues(local).transpose();
    if (kLocal) {
      value /= lagrange().LocalToJacobian(local).determinant();
    }
    return value;
  }

  Value GlobalToValue(Global const &global) const {
    Local local = lagrange().GlobalToLocal(global);
    return LobalToValue(local);
  }

  Global const &center() const {
    return gauss_ptr_->center();
  }
  Coeff const &coeff() const {
    return coeff_;
  }
  Coeff &coeff() {
    return coeff_;
  }
  Gauss const &gauss() const {
    return *gauss_ptr_;
  }
  Lagrange const &lagrange() const {
    return gauss().lagrange();
  }
  template <typename Callable>
  void Approximate(Callable &&global_to_value) {
    for (int ijk = 0; ijk < N; ++ijk) {
      auto &global = gauss_ptr_->GetGlobalCoord(ijk);
      coeff_.col(ijk) = global_to_value(global);  // value in physical space
      if (kLocal) {
        auto jacobian_det =
            gauss_ptr_->GetGlobalWeight(ijk) / gauss_ptr_->GetLocalWeight(ijk);
        coeff_.col(ijk) *= jacobian_det;  // value in parametric space
      }
    }
  }

  static Basis BuildInterpolationBasis() {
    auto line_x = typename Basis::LineX{ Gauss::GaussX::BuildPoints() };
    auto line_y = typename Basis::LineY{ Gauss::GaussY::BuildPoints() };
    auto line_z = typename Basis::LineZ{ Gauss::GaussZ::BuildPoints() };
    return Basis(line_x, line_y, line_z);
  }
};
template <std::floating_point Scalar, int kX, int kY, int kZ, int kC, bool kL>
typename Hexahedron<Scalar, kX, kY, kZ, kC, kL>::Basis const
Hexahedron<Scalar, kX, kY, kZ, kC, kL>::basis_ =
    Hexahedron<Scalar, kX, kY, kZ, kC, kL>::BuildInterpolationBasis();

}  // namespace polynomial
}  // namespace mini

#endif  // MINI_POLYNOMIAL_HEXAHEDRON_HPP_
