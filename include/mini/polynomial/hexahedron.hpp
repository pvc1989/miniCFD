//  Copyright 2023 PEI Weicheng
#ifndef MINI_POLYNOMIAL_HEXAHEDRON_HPP_
#define MINI_POLYNOMIAL_HEXAHEDRON_HPP_

#include <concepts>

#include <cmath>
#include <cstring>

#include <iostream>
#include <type_traits>
#include <utility>

#include "mini/algebra/eigen.hpp"
#include "mini/gauss/cell.hpp"
#include "mini/gauss/lobatto.hpp"
#include "mini/gauss/hexahedron.hpp"
#include "mini/basis/lagrange.hpp"

namespace mini {
namespace polynomial {

/**
 * @brief A vector-valued function interpolated on an given basis::lagrange::Hexahedron basis.
 * 
 * The interpolation nodes are collocated with quadrature points.
 * 
 * @tparam Gx  The quadrature rule in the 1st dimension.
 * @tparam Gy  The quadrature rule in the 2nd dimension.
 * @tparam Gz  The quadrature rule in the 3rd dimension.
 * @tparam kComponents the number of function components
 * @tparam kLocal in local (parametric) space or not
 */
template <class Gx, class Gy, class Gz, int kComponents, bool kLocal = false>
class Hexahedron {
 public:
  using GaussX = Gx;
  using GaussY = Gy;
  using GaussZ = Gz;
  using Gauss = gauss::Hexahedron<Gx, Gy, Gz>;
  using Scalar = typename Gauss::Scalar;
  using Local = typename Gauss::Local;
  using Global = typename Gauss::Global;
  using GaussBase = gauss::Cell<Scalar>;
  using Lagrange = typename Gauss::Lagrange;
  using Jacobian = typename Lagrange::Jacobian;
  static constexpr int Px = Gx::Q - 1;
  static constexpr int Py = Gy::Q - 1;
  static constexpr int Pz = Gz::Q - 1;
  static constexpr int P = std::max({Px, Py, Pz});
  using Basis = basis::lagrange::Hexahedron<Scalar, Px, Py, Pz>;
  static constexpr int N = Basis::N;
  static constexpr int K = kComponents;
  using Coeff = algebra::Matrix<Scalar, K, N>;
  using Value = algebra::Matrix<Scalar, K, 1>;
  using Mat1xN = algebra::Matrix<Scalar, 1, N>;
  using Mat3xN = algebra::Matrix<Scalar, 3, N>;

  using GaussOnLine = GaussX;

 private:
  static const Basis basis_;
  const Gauss *gauss_ptr_ = nullptr;
  Coeff coeff_;  // u^h(local) = coeff_ @ basis.GetValues(local)
  std::array<Mat3xN, N> basis_gradients_;

 public:
  explicit Hexahedron(const GaussBase &gauss)
      : gauss_ptr_(dynamic_cast<const Gauss *>(&gauss)) {
    for (int ijk = 0; ijk < N; ++ijk) {
      auto &grad = basis_gradients_[ijk];
      auto [i, j, k] = basis_.index(ijk);
      grad.row(0) = basis_.GetDerivatives(1, 0, 0, i, j, k);
      grad.row(1) = basis_.GetDerivatives(0, 1, 0, i, j, k);
      grad.row(2) = basis_.GetDerivatives(0, 0, 1, i, j, k);
      auto &local = gauss_ptr_->GetLocalCoord(ijk);
      Jacobian jacobian = lagrange().LocalToJacobian(local).transpose();
      LocalGradientsToGlobalGradients(jacobian, &grad);
    }
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
  Value GetValueOnGaussianPoint(int i) const {
    return coeff_.col(i);
  }
  Mat1xN GlobalToBasisValues(Global const &global) const {
    Local local = lagrange().GlobalToLocal(global);
    return basis_.GetValues(local);
  }
  Mat3xN GlobalToBasisGradients(Global const &global) const {
    Local local = lagrange().GlobalToLocal(global);
    Mat3xN grad;
    grad.row(0) = basis_.GetDerivatives(1, 0, 0, local);
    grad.row(1) = basis_.GetDerivatives(0, 1, 0, local);
    grad.row(2) = basis_.GetDerivatives(0, 0, 1, local);
    Jacobian jacobian = lagrange().LocalToJacobian(local).transpose();
    LocalGradientsToGlobalGradients(jacobian, &grad);
    return grad;
  }
  const Mat3xN &GetBasisGradientsOnGaussianPoint(int ijk) const {
    return basis_gradients_[ijk];
  }
  /**
   * @brief Convert the gradients in local coordinates to the gradients in global coordinates.
   * 
   * \f$ \begin{bmatrix}\partial\phi/\partial\xi\\ \partial\phi/\partial\eta\\ \cdots \end{bmatrix} = \begin{bmatrix}\partial x/\partial\xi & \partial y/\partial\xi & \cdots\\ \partial x/\partial\eta & \partial y/\partial\eta & \cdots\\ \cdots & \cdots & \cdots \end{bmatrix}\begin{bmatrix}\partial\phi/\partial x\\\partial\phi/\partial y\\ \cdots \end{bmatrix} \f$
   * 
   * @param jacobian the Jacobian matrix, which is the transpose of `geometry::Element::Jacobian`.
   * @param local the gradients in local coordinates
   */
  static void LocalGradientsToGlobalGradients(const Jacobian &jacobian,
      Mat3xN *local) {
    if (kLocal) return;
    Mat3xN global = jacobian.fullPivLu().solve(*local);
    *local = global;
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
  Basis const &basis() const {
    return basis_;
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
  const Scalar *GetCoeffFrom(const Scalar *input) {
    std::memcpy(coeff_.data(), input, sizeof(Scalar) * coeff_.size());
    return input + coeff_.size();
  }
  Scalar *WriteCoeffTo(Scalar *output) const {
    std::memcpy(output, coeff_.data(), sizeof(Scalar) * coeff_.size());
    return output + coeff_.size();
  }
  /**
   * @brief Add the given Coeff to the dofs corresponding to the given basis.
   * 
   * @param coeff the coeff to be added
   * @param output the beginning of all dofs
   */
  static void AddCoeffTo(Coeff const &coeff, Scalar *output) {
    for (int c = 0; c < N; ++c) {
      for (int r = 0; r < K; ++r) {
        *output++ += coeff(r, c);
      }
    }
  }
  /**
   * @brief Add the given Value to the dofs corresponding to the given basis.
   * 
   * @param value the value to be added
   * @param output the beginning of all dofs
   * @param i_basis the (0-based) index of basis
   */
  static void AddValueTo(Value const &value, Scalar *output, int i_basis) {
    output += K * i_basis;
    for (int r = 0; r < K; ++r) {
      *output++ += value[r];
    }
  }
  /**
   * @brief Multiply the given scale to the Value at the given address.
   * 
   * @param scale the scale to be multiplied
   * @param output the address of the value
   * @return the address of the next value
   */
  static Scalar *ScaleValueAt(double scale, Scalar *output) {
    for (int r = 0; r < K; ++r) {
      *output++ *= scale;
    }
    return output;
  }

  static Basis BuildInterpolationBasis() {
    auto line_x = typename Basis::LineX{ Gauss::GaussX::BuildPoints() };
    auto line_y = typename Basis::LineY{ Gauss::GaussY::BuildPoints() };
    auto line_z = typename Basis::LineZ{ Gauss::GaussZ::BuildPoints() };
    return Basis(line_x, line_y, line_z);
  }
};
template <class Gx, class Gy, class Gz, int kC, bool kL>
typename Hexahedron<Gx, Gy, Gz, kC, kL>::Basis const
Hexahedron<Gx, Gy, Gz, kC, kL>::basis_ =
    Hexahedron<Gx, Gy, Gz, kC, kL>::BuildInterpolationBasis();

}  // namespace polynomial
}  // namespace mini

#endif  // MINI_POLYNOMIAL_HEXAHEDRON_HPP_
