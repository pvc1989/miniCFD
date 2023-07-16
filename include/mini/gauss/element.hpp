//  Copyright 2023 PEI Weicheng
#ifndef MINI_GAUSS_ELEMENT_HPP_
#define MINI_GAUSS_ELEMENT_HPP_

#include "mini/lagrange/element.hpp"

namespace mini {
namespace gauss {

static constexpr int X{0}, Y{1}, Z{2};

/**
 * @brief Abstract numerical integrators on elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam kPhysDim  Dimension of the underlying physical space.
 * @tparam kCellDim  Dimension of the element as a manifold.
 */
template <std::floating_point Scalar, int kPhysDim, int kCellDim>
class Element {
 public:
  using Lagrange = lagrange::Element<Scalar, kPhysDim, kCellDim>;
  using Real = typename Lagrange::Real;
  using LocalCoord = typename Lagrange::LocalCoord;
  using GlobalCoord = typename Lagrange::GlobalCoord;
  using Jacobian = typename Lagrange::Jacobian;

  static constexpr int CellDim() { return kCellDim; }
  static constexpr int PhysDim() { return kPhysDim; }

  virtual ~Element() noexcept = default;

  /**
   * @brief Get the number of quadrature points on this element.
   * 
   * @return int  Number of quadrature points on this element.
   */
  virtual int CountQuadraturePoints() const = 0;

  /**
   * @brief Get the LocalCoord of the i-th quadrature point.
   * 
   * @param i  0-based index of the i-th quadrature point.
   * @return const LocalCoord &  LocalCoord of the i-th quadrature point.
   */
  virtual const LocalCoord &GetLocalCoord(int i) const = 0;

  /**
   * @brief Get the GlobalCoord of the i-th quadrature point.
   * 
   * @param i  0-based index of the i-th quadrature point.
   * @return const GlobalCoord &  GlobalCoord of the i-th quadrature point.
   */
  virtual const GlobalCoord &GetGlobalCoord(int i) const = 0;

  /**
   * @brief Get the local (without Jacobian) weight of the i-th quadrature point.
   * 
   * @param i  0-based index of the i-th quadrature point.
   * @return const Real &  Local weight of the i-th quadrature point.
   */
  virtual const Real &GetLocalWeight(int i) const = 0;

  /**
   * @brief Get the global (with Jacobian) weight of the i-th quadrature point.
   * 
   * @param i  0-based index of the i-th quadrature point.
   * @return const Real &  Global weight of the i-th quadrature point.
   */
  virtual const Real &GetGlobalWeight(int i) const = 0;

  /**
   * @brief Get a reference to the lagrange::Element object it uses for coordinate mapping.
   * 
   * @return const Lagrange &  Reference to the lagrange::Element object it uses for coordinate mapping.
   */
  virtual const Lagrange &lagrange() const = 0;

  int CountCorners() const {
    return lagrange().CountCorners();
  }
  int CountNodes() const {
    return lagrange().CountNodes();
  }
  const GlobalCoord &center() const {
    return lagrange().center();
  }
  GlobalCoord LocalToGlobal(const LocalCoord &local) const {
    return lagrange().LocalToGlobal(local);
  }
  Jacobian LocalToJacobian(const LocalCoord &local) const {
    return lagrange().LocalToJacobian(local);
  }

 protected:
  virtual GlobalCoord &GetGlobalCoord(int i) = 0;
  virtual Real &GetGlobalWeight(int i) = 0;
  Real BuildQuadraturePoints() {
    Real sum = 0.0;
    for (int i = 0, n = CountQuadraturePoints(); i < n; ++i) {
      auto mat_j = lagrange().LocalToJacobian(GetLocalCoord(i));
      auto det_j = this->CellDim() < this->PhysDim()
          ? std::sqrt((mat_j.transpose() * mat_j).determinant())
          : mat_j.determinant();
      GetGlobalWeight(i) = GetLocalWeight(i) * det_j;
      sum += GetGlobalWeight(i);
      GetGlobaCoord(i) = lagrange().LocalToGlobal(lagrange().GetLocalCoord(i));
    }
    return sum;
  }
};

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_ELEMENT_HPP_
