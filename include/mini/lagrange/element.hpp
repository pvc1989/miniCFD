//  Copyright 2023 PEI Weicheng
#ifndef MINI_LAGRANGE_ELEMENT_HPP_
#define MINI_LAGRANGE_ELEMENT_HPP_

#include "mini/algebra/eigen.hpp"

namespace mini {
namespace lagrange {

static constexpr int X{0}, Y{1}, Z{2};

/**
 * @brief Abstract coordinate map on surface elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam kPhysDim  Dimension of the underlying physical space.
 * @tparam kCellDim  Dimension of the element as a manifold.
 */
template <std::floating_point Scalar, int kPhysDim, int kCellDim>
class Element {
 public:
  using Real = Scalar;
  using LocalCoord = algebra::Matrix<Scalar, kCellDim, 1>;
  using GlobalCoord = algebra::Matrix<Scalar, kPhysDim, 1>;
  using Jacobian = algebra::Matrix<Scalar, kPhysDim, kCellDim>;

  static constexpr int CellDim() { return kCellDim; }
  static constexpr int PhysDim() { return kPhysDim; }

  virtual ~Element() noexcept = default;
  virtual std::vector<Scalar> LocalToShapeFunctions(const LocalCoord &) const = 0;
  virtual std::vector<LocalCoord> LocalToShapeGradients(const LocalCoord &) const = 0;
  virtual GlobalCoord LocalToGlobal(const LocalCoord &) const = 0;
  virtual Jacobian LocalToJacobian(const LocalCoord &) const = 0;
  virtual int CountCorners() const = 0;
  virtual int CountNodes() const = 0;
  virtual const LocalCoord &GetLocalCoord(int i) const = 0;
  virtual const GlobalCoord &GetGlobalCoord(int i) const = 0;
  virtual const GlobalCoord &center() const = 0;

};

}  // namespace lagrange
}  // namespace mini

#endif  // MINI_LAGRANGE_ELEMENT_HPP_
