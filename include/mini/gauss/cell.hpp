//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_GAUSS_CELL_HPP_
#define MINI_GAUSS_CELL_HPP_

#include <concepts>

#include "mini/lagrange/cell.hpp"
#include "mini/gauss/element.hpp"
#include "mini/gauss/function.hpp"

namespace mini {
namespace gauss {

/**
 * @brief Abstract numerical integrators on volume elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Cell {

 public:
  using Lagrange = lagrange::Cell<Scalar>;
  using Real = typename Lagrange::Real;
  using LocalCoord = typename Lagrange::LocalCoord;
  using GlobalCoord = typename Lagrange::GlobalCoord;
  using Jacobian = typename Lagrange::Jacobian;

  virtual ~Cell() noexcept = default;
  virtual int CountQuadraturePoints() const = 0;
  virtual const LocalCoord &GetLocalCoord(int i) const = 0;
  virtual const GlobalCoord &GetGlobalCoord(int i) const = 0;
  virtual Real const &GetLocalWeight(int i) const = 0;
  virtual Real const &GetGlobalWeight(int i) const = 0;
  virtual Real volume() const = 0;

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

  GlobalCoord LocalToGlobal(Scalar x_local, Scalar y_local, Scalar z_local)
      const {
    return lagrange().LocalToGlobal(x_local, y_local, z_local);
  }
  GlobalCoord LocalToGlobal(const LocalCoord &xyz) const {
    return LocalToGlobal(xyz[X], xyz[Y], xyz[Z]);
  }

  Jacobian LocalToJacobian(Scalar x_local, Scalar y_local, Scalar z_local)
      const {
    return lagrange().LocalToJacobian(x_local, y_local, z_local);
  }
  Jacobian LocalToJacobian(const LocalCoord &xyz) const {
    return LocalToJacobian(xyz[X], xyz[Y], xyz[Z]);
  }

  LocalCoord GlobalToLocal(Scalar x_global, Scalar y_global, Scalar z_global)
      const {
    return lagrange().GlobalToLocal(x_global, y_global, z_global);
  }
  LocalCoord GlobalToLocal(const GlobalCoord &xyz) const {
    return GlobalToLocal(xyz[X], xyz[Y], xyz[Z]);
  }

  static constexpr int CellDim() {
    return 3;
  }
  static constexpr int PhysDim() {
    return 3;
  }
};

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_CELL_HPP_
