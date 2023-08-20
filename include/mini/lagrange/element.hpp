//  Copyright 2023 PEI Weicheng
#ifndef MINI_LAGRANGE_ELEMENT_HPP_
#define MINI_LAGRANGE_ELEMENT_HPP_

#include <concepts>

#include <cassert>

#include <initializer_list>
#include <vector>

#include "mini/algebra/eigen.hpp"

namespace mini {
namespace lagrange {

static constexpr int X{0}, Y{1}, Z{2};
static constexpr int A{0}, B{1}, C{2}, D{3};

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
  using Local = algebra::Matrix<Scalar, kCellDim, 1>;
  using Global = algebra::Matrix<Scalar, kPhysDim, 1>;
  using Jacobian = algebra::Matrix<Scalar, kPhysDim, kCellDim>;

  static constexpr int CellDim() { return kCellDim; }
  static constexpr int PhysDim() { return kPhysDim; }

  virtual ~Element() noexcept = default;
  virtual std::vector<Scalar> LocalToShapeFunctions(const Local &) const = 0;
  virtual std::vector<Local> LocalToShapeGradients(const Local &) const = 0;
  virtual Global LocalToGlobal(const Local &) const = 0;
  virtual Jacobian LocalToJacobian(const Local &) const = 0;
  virtual int CountCorners() const = 0;
  virtual int CountNodes() const = 0;
  virtual const Local &GetLocalCoord(int i) const = 0;
  virtual const Global &GetGlobalCoord(int i) const = 0;
  virtual const Global &center() const = 0;

 protected:
  virtual void _BuildCenter() = 0;
  Global &_GetGlobalCoord(int i) {
    const Global &global
        = const_cast<const Element *>(this)->GetGlobalCoord(i);
    return const_cast<Global &>(global);
  }
};

template <typename Element>
static void _Build(Element *element,
    std::initializer_list<typename Element::Global> il) {
  assert(il.size() == element->CountNodes());
  auto p = il.begin();
  for (int i = 0, n = element->CountNodes(); i < n; ++i) {
    element->_GetGlobalCoord(i) = p[i];
  }
  element->_BuildCenter();
}

}  // namespace lagrange
}  // namespace mini

#endif  // MINI_LAGRANGE_ELEMENT_HPP_
