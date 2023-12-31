//  Copyright 2023 PEI Weicheng
#ifndef MINI_POLYNOMIAL_HEXAHEDRON_HPP_
#define MINI_POLYNOMIAL_HEXAHEDRON_HPP_

#include <concepts>

#include <cmath>
#include <cstring>

#include <iostream>
#include <tuple>
#include <type_traits>
#include <utility>

#include "mini/algebra/eigen.hpp"
#include "mini/gauss/cell.hpp"
#include "mini/gauss/lobatto.hpp"
#include "mini/gauss/hexahedron.hpp"
#include "mini/geometry/element.hpp"
#include "mini/basis/lagrange.hpp"
#include "mini/constant/index.hpp"

namespace mini {
namespace polynomial {

using namespace mini::constant::index;

/**
 * @brief A vector-valued function interpolated on an given basis::lagrange::Hexahedron basis.
 * 
 * The interpolation nodes are collocated with quadrature points.
 * 
 * @tparam Gx  The quadrature rule in the 1st dimension.
 * @tparam Gy  The quadrature rule in the 2nd dimension.
 * @tparam Gz  The quadrature rule in the 3rd dimension.
 * @tparam kC  The number of function components.
 * @tparam kL  Formulate in local (parametric) space or not.
 */
template <class Gx, class Gy, class Gz, int kC, bool kL = false>
class Hexahedron {
 public:
  static constexpr bool kLocal = kL;
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
  static constexpr int K = kC;
  using Coeff = algebra::Matrix<Scalar, K, N>;
  using Value = algebra::Matrix<Scalar, K, 1>;
  using Mat1xN = algebra::Matrix<Scalar, 1, N>;
  using Mat3xN = algebra::Matrix<Scalar, 3, N>;
  using Mat6xN = algebra::Matrix<Scalar, 3, N>;
  using Gradient = algebra::Matrix<Scalar, 3, K>;
  using Hessian = algebra::Matrix<Scalar, 6, K>;

  using GaussOnLine = GaussX;

 private:
  const Gauss *gauss_ptr_ = nullptr;
  Coeff coeff_;  // u^h(local) = coeff_ @ basis.GetValues(local)

  struct E { };
  /* \f$ \det(\mathbf{J}) \f$ */
  [[no_unique_address]] std::conditional_t<kLocal, std::array<Scalar, N>, E>
      jacobian_det_;
  /* \f$ \det(\mathbf{J})\,\mathbf{J}^{-1} \f$ */
  [[no_unique_address]] std::conditional_t<kLocal, std::array<Jacobian, N>, E>
      jacobian_det_inv_;
  /* \f$ \begin{bmatrix}\partial_{\xi}\\ \partial_{\eta}\\ \partial_{\zeta} \end{bmatrix}\det(\mathbf{J}) \f$ */
  [[no_unique_address]] std::conditional_t<kLocal, std::array<Local, N>, E>
      jacobian_det_grad_;
  [[no_unique_address]] std::conditional_t<kLocal, E, std::array<Mat3xN, N>>
      basis_global_gradients_;

  static void CheckSize() {
    constexpr size_t large_member_size = kLocal
        ? sizeof(std::array<Scalar, N>) + sizeof(std::array<Jacobian, N>)
            + sizeof(std::array<Local, N>)
        : sizeof(std::array<Mat3xN, N>);
    constexpr size_t all_member_size = large_member_size
        + sizeof(gauss_ptr_) + sizeof(coeff_);
    static_assert(sizeof(Hexahedron) >= all_member_size);
    static_assert(sizeof(Hexahedron) <= all_member_size + 16);
  }

  static const Basis basis_;
  static Basis BuildInterpolationBasis() {
    CheckSize();
    auto line_x = typename Basis::LineX{ Gauss::GaussX::BuildPoints() };
    auto line_y = typename Basis::LineY{ Gauss::GaussY::BuildPoints() };
    auto line_z = typename Basis::LineZ{ Gauss::GaussZ::BuildPoints() };
    return Basis(line_x, line_y, line_z);
  }

  static const std::array<Mat3xN, N> basis_local_gradients_;
  static std::array<Mat3xN, N> BuildBasisLocalGradients() {
    std::array<Mat3xN, N> gradients;
    auto basis = BuildInterpolationBasis();
    for (int ijk = 0; ijk < N; ++ijk) {
      auto &grad = gradients[ijk];
      auto [i, j, k] = basis.index(ijk);
      grad.row(0) = basis.GetDerivatives(1, 0, 0, i, j, k);
      grad.row(1) = basis.GetDerivatives(0, 1, 0, i, j, k);
      grad.row(2) = basis.GetDerivatives(0, 0, 1, i, j, k);
    }
    return gradients;
  }
  static const std::array<Mat6xN, N> basis_local_hessians_;
  static std::array<Mat6xN, N> BuildBasisLocalHessians() {
    std::array<Mat6xN, N> hessians;
    auto basis = BuildInterpolationBasis();
    for (int ijk = 0; ijk < N; ++ijk) {
      auto &hess = hessians[ijk];
      auto [i, j, k] = basis.index(ijk);
      hess.row(XX) = basis.GetDerivatives(2, 0, 0, i, j, k);
      hess.row(XY) = basis.GetDerivatives(1, 1, 0, i, j, k);
      hess.row(XZ) = basis.GetDerivatives(1, 0, 1, i, j, k);
      hess.row(YY) = basis.GetDerivatives(0, 2, 0, i, j, k);
      hess.row(YZ) = basis.GetDerivatives(0, 1, 1, i, j, k);
      hess.row(ZZ) = basis.GetDerivatives(0, 0, 2, i, j, k);
    }
    return hessians;
  }

 public:
  explicit Hexahedron(const GaussBase &gauss) requires (kLocal)
      : gauss_ptr_(dynamic_cast<const Gauss *>(&gauss)) {
    for (int ijk = 0; ijk < N; ++ijk) {
      auto &local = gauss_ptr_->GetLocalCoord(ijk);
      Jacobian jacobian = lagrange().LocalToJacobian(local);
      jacobian_det_[ijk] = jacobian.determinant();
      jacobian_det_inv_[ijk] = jacobian_det_[ijk] * jacobian.inverse();
      jacobian_det_grad_[ijk]
          = lagrange().LocalToJacobianDeterminantGradient(local);
    }
  }
  explicit Hexahedron(const GaussBase &gauss) requires (!kLocal)
      : gauss_ptr_(dynamic_cast<const Gauss *>(&gauss)) {
    for (int ijk = 0; ijk < N; ++ijk) {
      auto &local = gauss_ptr_->GetLocalCoord(ijk);
      Jacobian jacobian = lagrange().LocalToJacobian(local);
      basis_global_gradients_[ijk] = LocalGradientsToGlobalGradients(
          jacobian, basis_local_gradients_[ijk]);
    }
  }
  Hexahedron() = default;
  Hexahedron(const Hexahedron &) = default;
  Hexahedron(Hexahedron &&) noexcept = default;
  Hexahedron &operator=(const Hexahedron &) = default;
  Hexahedron &operator=(Hexahedron &&) noexcept = default;
  ~Hexahedron() noexcept = default;

  Value LobalToValue(Local const &local) const requires (kLocal) {
    Value value = coeff_ * basis_.GetValues(local).transpose();
    value /= lagrange().LocalToJacobian(local).determinant();
    return value;
  }
  Value LobalToValue(Local const &local) const requires (!kLocal) {
    return coeff_ * basis_.GetValues(local).transpose();
  }
  void LocalToGlobalAndValue(Local const &local,
      Global *global, Value *value) const {
    *global = gauss().lagrange().LocalToGlobal(local);
    *value = LobalToValue(local);
  }

  Value GlobalToValue(Global const &global) const {
    Local local = lagrange().GlobalToLocal(global);
    return LobalToValue(local);
  }
  /**
   * @brief Get the value of \f$ u(x,y,z) \equiv \det(\mathbf{J})^{-1}\,U(\xi,\eta,\zeta) \f$ at a Gaussian point.
   * 
   * This version is compiled only if `kLocal` is `true`.
   * 
   * @param ijk the index of the Gaussian point
   * @return Value the value \f$ u(x_i,y_i,z_i) \f$
   */
  Value GetValue(int i) const requires (kLocal) {
    return coeff_.col(i) / jacobian_det_[i];
  }
  /**
   * @brief Get the value of \f$ u(x,y,z) \f$ at a Gaussian point.
   * 
   * This version is compiled only if `kLocal` is `false`.
   * 
   * @param ijk the index of the Gaussian point
   * @return Value the value \f$ u(x_i,y_i,z_i) \f$
   */
  Value GetValue(int i) const requires (!kLocal) {
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
    Jacobian jacobian = lagrange().LocalToJacobian(local);
    return LocalGradientsToGlobalGradients(jacobian, grad);
  }
  /**
   * @brief Get the local gradients of basis at a Gaussian point.
   * 
   * This version is compiled only if `kLocal` is `true`.
   * 
   * @param ijk the index of the Gaussian point
   * @return const Mat3xN& the local gradients of basis
   */
  const Mat3xN &GetBasisGradients(int ijk) const requires (kLocal) {
    return basis_local_gradients_[ijk];
  }
  /**
   * @brief Get the global gradients of basis at a Gaussian point.
   * 
   * This version is compiled only if `kLocal` is `false`.
   * 
   * @param ijk the index of the Gaussian point
   * @return const Mat3xN& the global gradients of basis
   */
  const Mat3xN &GetBasisGradients(int ijk) const requires (!kLocal) {
    return basis_global_gradients_[ijk];
  }
  /**
   * @brief Get \f$ \begin{bmatrix}\partial_{\xi}\\ \partial_{\eta}\\ \cdots \end{bmatrix} U \f$ at a Gaussian point.
   * 
   */
  Gradient GetLocalGradient(int ijk) const requires (kLocal) {
    Gradient value_grad; value_grad.setZero();
    Mat3xN const &basis_grads = GetBasisGradients(ijk);
    for (int abc = 0; abc < N; ++abc) {
      value_grad += basis_grads.col(abc) * coeff_.col(abc).transpose();
    }
    return value_grad;
  }
  /**
   * @brief Get \f$ \begin{bmatrix}\partial_{x}\\ \partial_{y}\\ \cdots \end{bmatrix} u \f$ at a Gaussian point.
   * 
   */
  Gradient GetGlobalGradient(int ijk) const requires (kLocal) {
    auto value_grad = GetLocalGradient(ijk);
    value_grad -= jacobian_det_grad_[ijk] * GetValue(ijk).transpose();
    auto jacobian_det = jacobian_det_[ijk];
    value_grad /= (jacobian_det * jacobian_det);
    return GetJacobianAssociated(ijk) * value_grad;
  }
  /**
   * @brief Get the local Hessians of basis at a Gaussian point.
   * 
   * This version is compiled only if `kLocal` is `true`.
   * 
   * @param ijk the index of the Gaussian point
   * @return const Mat6xN& the local Hessians of basis
   */
  const Mat6xN &GetBasisHessians(int ijk) const requires (kLocal) {
    return basis_local_hessians_[ijk];
  }
  /**
   * @brief Get \f$ \begin{bmatrix}\partial_{\xi}\partial_{\xi}\\ \partial_{\xi}\partial_{\eta}\\ \cdots \end{bmatrix} U \f$ at a Gaussian point.
   * 
   */
  Hessian GetLocalHessian(int ijk) const requires (kLocal) {
    Hessian value_hess; value_hess.setZero();
    Mat6xN const &basis_hess = GetBasisHessians(ijk);
    for (int abc = 0; abc < N; ++abc) {
      value_hess += basis_hess.col(abc) * coeff_.col(abc).transpose();
    }
    return value_hess;
  }
  /**
   * @brief Get \f$ \begin{bmatrix}\partial_{x}\partial_{x}\\ \partial_{x}\partial_{y}\\ \cdots \end{bmatrix} u \f$ at a Gaussian point.
   * 
   */
  Hessian GetGlobalHessian(int ijk) const requires (kLocal) {
    Hessian value_hess = GetLocalHessian(ijk);
    value_hess -= jacobian_det_grad_[ijk] * GetValue(ijk).transpose();
    auto jacobian_det = jacobian_det_[ijk];
    value_hess /= (jacobian_det * jacobian_det);
    return GetJacobianAssociated(ijk) * value_hess;
  }

  /**
   * @brief Convert the gradients in local coordinates to the gradients in global coordinates.
   * 
   * \f$ \begin{bmatrix}\partial_{\xi}\,\phi\\\partial_{\eta}\,\phi\\\partial_{\zeta}\,\phi\end{bmatrix}=\mathbf{J}\begin{bmatrix}\partial_{x}\,\phi\\\partial_{y}\,\phi\\\partial_{z}\,\phi\end{bmatrix} \f$, in which \f$ \mathbf{J}=\begin{bmatrix}\partial_{\xi}\\\partial_{\eta}\\\partial_{\zeta}\end{bmatrix}\begin{bmatrix}x & y & z\end{bmatrix} \f$ is `geometry::Element::Jacobian`.
   * 
   * @param jacobian the Jacobian matrix of the type `geometry::Element::Jacobian`.
   * @param local_grad the gradients in local coordinates
   * @return Mat3xN the gradients in global coordinates
   */
  static Mat3xN LocalGradientsToGlobalGradients(const Jacobian &jacobian,
      Mat3xN const &local_grad) {
    return jacobian.fullPivLu().solve(local_grad);
  }

  /**
   * @brief Convert a flux matrix from global to local at a given Gaussian point.
   * 
   * \f$ \begin{bmatrix}F^{\xi} & F^{\eta} & F^{\zeta}\end{bmatrix}=\begin{bmatrix}f^{x} & f^{y} & f^{z}\end{bmatrix}\mathbf{J}^{*} \f$, in which \f$ \mathbf{J}^{*} = \det(\mathbf{J}) \begin{bmatrix}\partial_{x}\\\partial_{y}\\\partial_{z}\end{bmatrix}\begin{bmatrix}\xi & \eta & \zeta\end{bmatrix} \f$ is returned by `Hexahedron::GetJacobianAssociated`.
   * 
   * @tparam FluxMatrix a matrix type which has 3 columns
   * @param global_flux the global flux
   * @param ijk the index of the Gaussian point
   * @return FluxMatrix the local flux
   */
  template <class FluxMatrix>
  FluxMatrix GlobalFluxToLocalFlux(const FluxMatrix &global_flux, int ijk) const
      requires (kLocal) {
    FluxMatrix local_flux = global_flux * GetJacobianAssociated(ijk);
    return local_flux;
  }

  /**
   * @brief Get the associated matrix of the Jacobian at a given Gaussian point.
   * 
   * \f$ \mathbf{J}^{*}=\det(\mathbf{J})\,\mathbf{J}^{-1} \f$, in which \f$ \mathbf{J}^{-1}=\begin{bmatrix}\partial_{x}\\\partial_{y}\\\partial_{z}\end{bmatrix}\begin{bmatrix}\xi & \eta & \zeta\end{bmatrix} \f$ is the inverse of `geometry::Element::Jacobian`.
   * 
   * @param ijk the index of the Gaussian point
   * @return Jacobian const& the associated matrix of \f$ \mathbf{J} \f$.
   */
  Jacobian const &GetJacobianAssociated(int ijk) const
      requires (kLocal) {
    return jacobian_det_inv_[ijk];
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
  void Approximate(Callable &&global_to_value) requires (kLocal) {
    for (int ijk = 0; ijk < N; ++ijk) {
      auto &global = gauss_ptr_->GetGlobalCoord(ijk);
      coeff_.col(ijk) = global_to_value(global);  // value in physical space
      coeff_.col(ijk) *= jacobian_det_[ijk];  // value in parametric space
    }
  }
  template <typename Callable>
  void Approximate(Callable &&global_to_value) requires (!kLocal) {
    for (int ijk = 0; ijk < N; ++ijk) {
      auto &global = gauss_ptr_->GetGlobalCoord(ijk);
      coeff_.col(ijk) = global_to_value(global);  // value in physical space
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
  static void MinusCoeff(Coeff const &coeff, Scalar *output) {
    for (int c = 0; c < N; ++c) {
      for (int r = 0; r < K; ++r) {
        *output++ -= coeff(r, c);
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
    assert(0 <= i_basis && i_basis < N);
    output += K * i_basis;
    for (int r = 0; r < K; ++r) {
      *output++ += value[r];
    }
  }
  static void MinusValue(Value const &value, Scalar *output, int i_basis) {
    assert(0 <= i_basis && i_basis < N);
    output += K * i_basis;
    for (int r = 0; r < K; ++r) {
      *output++ -= value[r];
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

  int FindFaceId(Global const &face_center) const {
    int i_face;
    auto almost_equal = [&face_center](Global point) {
      point -= face_center;
      return point.norm() < 1e-10;
    };
    if (almost_equal(lagrange().LocalToGlobal(0, 0, -1))) { i_face = 0; }
    else if (almost_equal(lagrange().LocalToGlobal(0, -1, 0))) { i_face = 1; }
    else if (almost_equal(lagrange().LocalToGlobal(+1, 0, 0))) { i_face = 2; }
    else if (almost_equal(lagrange().LocalToGlobal(0, +1, 0))) { i_face = 3; }
    else if (almost_equal(lagrange().LocalToGlobal(-1, 0, 0))) { i_face = 4; }
    else if (almost_equal(lagrange().LocalToGlobal(0, 0, +1))) { i_face = 5; }
    else { assert(false); }
    return i_face;
  }
  std::vector<int> FindCollinearPoints(Global const &global, int i_face) const {
    std::vector<int> indices;
    using mini::geometry::X;
    using mini::geometry::Y;
    using mini::geometry::Z;
    auto local = lagrange().GlobalToLocal(global);
    int i, j, k;
    auto almost_equal = [](Scalar x, Scalar y) {
      return std::abs(x - y) < 1e-10;
    };
    switch (i_face) {
    case 0:
      assert(almost_equal(local[Z], -1));
      for (i = 0; i < GaussX::Q; ++i) {
        if (almost_equal(local[X], GaussX::points[i])) {
          break;
        }
      }
      for (j = 0; j < GaussY::Q; ++j) {
        if (almost_equal(local[Y], GaussY::points[j])) {
          break;
        }
      }
      for (k = 0; k < GaussZ::Q; ++k) {
        indices.push_back(basis().index(i, j, k));
      }
      break;
    case 1:
      assert(almost_equal(local[Y], -1));
      for (i = 0; i < GaussX::Q; ++i) {
        if (almost_equal(local[X], GaussX::points[i])) {
          break;
        }
      }
      for (k = 0; k < GaussZ::Q; ++k) {
        if (almost_equal(local[Z], GaussZ::points[k])) {
          break;
        }
      }
      for (j = 0; j < GaussY::Q; ++j) {
        indices.push_back(basis().index(i, j, k));
      }
      break;
    case 2:
      assert(almost_equal(local[X], +1));
      for (j = 0; j < GaussY::Q; ++j) {
        if (almost_equal(local[Y], GaussY::points[j])) {
          break;
        }
      }
      for (k = 0; k < GaussZ::Q; ++k) {
        if (almost_equal(local[Z], GaussZ::points[k])) {
          break;
        }
      }
      for (i = 0; i < GaussX::Q; ++i) {
        indices.push_back(basis().index(i, j, k));
      }
      break;
    case 3:
      assert(almost_equal(local[Y], +1));
      for (i = 0; i < GaussX::Q; ++i) {
        if (almost_equal(local[X], GaussX::points[i])) {
          break;
        }
      }
      for (k = 0; k < GaussZ::Q; ++k) {
        if (almost_equal(local[Z], GaussZ::points[k])) {
          break;
        }
      }
      for (j = 0; j < GaussY::Q; ++j) {
        indices.push_back(basis().index(i, j, k));
      }
      break;
    case 4:
      assert(almost_equal(local[X], -1));
      for (j = 0; j < GaussY::Q; ++j) {
        if (almost_equal(local[Y], GaussY::points[j])) {
          break;
        }
      }
      for (k = 0; k < GaussZ::Q; ++k) {
        if (almost_equal(local[Z], GaussZ::points[k])) {
          break;
        }
      }
      for (i = 0; i < GaussX::Q; ++i) {
        indices.push_back(basis().index(i, j, k));
      }
      break;
    case 5:
      assert(almost_equal(local[Z], +1));
      for (i = 0; i < GaussX::Q; ++i) {
        if (almost_equal(local[X], GaussX::points[i])) {
          break;
        }
      }
      for (j = 0; j < GaussY::Q; ++j) {
        if (almost_equal(local[Y], GaussY::points[j])) {
          break;
        }
      }
      for (k = 0; k < GaussZ::Q; ++k) {
        indices.push_back(basis().index(i, j, k));
      }
      break;
    default: assert(false);
    }
    return indices;
  }
  std::tuple<int, int, int> FindCollinearIndex(Global const &global, int i_face) const {
    return FindCollinearIndexByGlobal(global, i_face);
  }
  std::tuple<int, int, int> FindCollinearIndexByGlobal(Global const &global, int i_face) const {
    int i{-1}, j{-1}, k{-1};
    using mini::geometry::X;
    using mini::geometry::Y;
    using mini::geometry::Z;
    Global global_temp;
    bool done = false;
    switch (i_face) {
    case 0:
      for (i = 0; i < GaussX::Q; ++i) {
        for (j = 0; j < GaussY::Q; ++j) {
          global_temp = lagrange().LocalToGlobal(
              GaussX::points[i], GaussY::points[j], -1);
          global_temp -= global;
          if (global_temp.norm() < 1e-8) {
            done = true;
            break;
          }
        }
        if (done) {
          break;
        }
      }
      break;
    case 1:
      for (i = 0; i < GaussX::Q; ++i) {
        for (k = 0; k < GaussZ::Q; ++k) {
          global_temp = lagrange().LocalToGlobal(
              GaussX::points[i], -1, GaussZ::points[k]);
          global_temp -= global;
          if (global_temp.norm() < 1e-8) {
            done = true;
            break;
          }
        }
        if (done) {
          break;
        }
      }
      break;
    case 2:
      for (j = 0; j < GaussY::Q; ++j) {
        for (k = 0; k < GaussZ::Q; ++k) {
          global_temp = lagrange().LocalToGlobal(
              +1, GaussY::points[j], GaussZ::points[k]);
          global_temp -= global;
          if (global_temp.norm() < 1e-8) {
            done = true;
            break;
          }
        }
        if (done) {
          break;
        }
      }
      break;
    case 3:
      for (i = 0; i < GaussX::Q; ++i) {
        for (k = 0; k < GaussZ::Q; ++k) {
          global_temp = lagrange().LocalToGlobal(
              GaussX::points[i], +1, GaussZ::points[k]);
          global_temp -= global;
          if (global_temp.norm() < 1e-8) {
            done = true;
            break;
          }
        }
        if (done) {
          break;
        }
      }
      break;
    case 4:
      for (j = 0; j < GaussY::Q; ++j) {
        for (k = 0; k < GaussZ::Q; ++k) {
          global_temp = lagrange().LocalToGlobal(
              -1, GaussY::points[j], GaussZ::points[k]);
          global_temp -= global;
          if (global_temp.norm() < 1e-8) {
            done = true;
            break;
          }
        }
        if (done) {
          break;
        }
      }
      break;
    case 5:
      for (i = 0; i < GaussX::Q; ++i) {
        for (j = 0; j < GaussY::Q; ++j) {
          global_temp = lagrange().LocalToGlobal(
              GaussX::points[i], GaussY::points[j], +1);
          global_temp -= global;
          if (global_temp.norm() < 1e-8) {
            done = true;
            break;
          }
        }
        if (done) {
          break;
        }
      }
      break;
    default: assert(false);
    }
    return std::make_tuple(i, j, k);
  }
  std::tuple<int, int, int> FindCollinearIndexByLocal(Global const &global, int i_face) const {
    int i{-1}, j{-1}, k{-1};
    using mini::geometry::X;
    using mini::geometry::Y;
    using mini::geometry::Z;
    Local local_hint(0, 0, 0);
    switch (i_face) {
    case 0: local_hint[Z] = -1; break;
    case 1: local_hint[Y] = -1; break;
    case 2: local_hint[X] = +1; break;
    case 3: local_hint[Y] = +1; break;
    case 4: local_hint[X] = -1; break;
    case 5: local_hint[Z] = +1; break;
    default: assert(false);
    }
    auto local = lagrange().GlobalToLocal(global, local_hint);
    auto almost_equal = [](Scalar x, Scalar y) {
      return std::abs(x - y) < 1e-10;
    };
    switch (i_face) {
    case 0:
      assert(almost_equal(local[Z], -1));
      for (i = 0; i < GaussX::Q; ++i) {
        if (almost_equal(local[X], GaussX::points[i])) {
          break;
        }
      }
      for (j = 0; j < GaussY::Q; ++j) {
        if (almost_equal(local[Y], GaussY::points[j])) {
          break;
        }
      }
      break;
    case 1:
      assert(almost_equal(local[Y], -1));
      for (i = 0; i < GaussX::Q; ++i) {
        if (almost_equal(local[X], GaussX::points[i])) {
          break;
        }
      }
      for (k = 0; k < GaussZ::Q; ++k) {
        if (almost_equal(local[Z], GaussZ::points[k])) {
          break;
        }
      }
      break;
    case 2:
      assert(almost_equal(local[X], +1));
      for (j = 0; j < GaussY::Q; ++j) {
        if (almost_equal(local[Y], GaussY::points[j])) {
          break;
        }
      }
      for (k = 0; k < GaussZ::Q; ++k) {
        if (almost_equal(local[Z], GaussZ::points[k])) {
          break;
        }
      }
      break;
    case 3:
      assert(almost_equal(local[Y], +1));
      for (i = 0; i < GaussX::Q; ++i) {
        if (almost_equal(local[X], GaussX::points[i])) {
          break;
        }
      }
      for (k = 0; k < GaussZ::Q; ++k) {
        if (almost_equal(local[Z], GaussZ::points[k])) {
          break;
        }
      }
      break;
    case 4:
      assert(almost_equal(local[X], -1));
      for (j = 0; j < GaussY::Q; ++j) {
        if (almost_equal(local[Y], GaussY::points[j])) {
          break;
        }
      }
      for (k = 0; k < GaussZ::Q; ++k) {
        if (almost_equal(local[Z], GaussZ::points[k])) {
          break;
        }
      }
      break;
    case 5:
      assert(almost_equal(local[Z], +1));
      for (i = 0; i < GaussX::Q; ++i) {
        if (almost_equal(local[X], GaussX::points[i])) {
          break;
        }
      }
      for (j = 0; j < GaussY::Q; ++j) {
        if (almost_equal(local[Y], GaussY::points[j])) {
          break;
        }
      }
      break;
    default: assert(false);
    }
    return std::make_tuple(i, j, k);
  }
};
template <class Gx, class Gy, class Gz, int kC, bool kL>
typename Hexahedron<Gx, Gy, Gz, kC, kL>::Basis const
Hexahedron<Gx, Gy, Gz, kC, kL>::basis_ =
    Hexahedron<Gx, Gy, Gz, kC, kL>::BuildInterpolationBasis();

template <class Gx, class Gy, class Gz, int kC, bool kL>
std::array<typename Hexahedron<Gx, Gy, Gz, kC, kL>::Mat3xN,
                    Hexahedron<Gx, Gy, Gz, kC, kL>::N> const
Hexahedron<Gx, Gy, Gz, kC, kL>::basis_local_gradients_ =
    Hexahedron<Gx, Gy, Gz, kC, kL>::BuildBasisLocalGradients();

template <class Gx, class Gy, class Gz, int kC, bool kL>
std::array<typename Hexahedron<Gx, Gy, Gz, kC, kL>::Mat6xN,
                    Hexahedron<Gx, Gy, Gz, kC, kL>::N> const
Hexahedron<Gx, Gy, Gz, kC, kL>::basis_local_hessians_ =
    Hexahedron<Gx, Gy, Gz, kC, kL>::BuildBasisLocalHessians();

}  // namespace polynomial
}  // namespace mini

#endif  // MINI_POLYNOMIAL_HEXAHEDRON_HPP_
