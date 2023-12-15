//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_GEOMETRY_CELL_HPP_
#define MINI_GEOMETRY_CELL_HPP_

#include <concepts>

#include <algorithm>
#include <vector>

#include "mini/geometry/element.hpp"

namespace mini {
namespace geometry {

static constexpr int XX{0}, XY{1}, XZ{2}, YY{3}, YZ{4}, ZZ{5};

/**
 * @brief Abstract coordinate map on volume elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Cell : public Element<Scalar, 3, 3> {
  using Base = Element<Scalar, 3, 3>;

 public:
  using Real = typename Base::Real;
  using Local = typename Base::Local;
  using Global = typename Base::Global;
  using Jacobian = typename Base::Jacobian;

  /**
   * @brief The type of (geometric) Hessian matrix of a scalar function \f$ f \f$, which is defined as \f$ \begin{bmatrix}\partial_{\xi}\partial_{\xi}f & \partial_{\xi}\partial_{\eta}f & \partial_{\xi}\partial_{\zeta}f\\ \partial_{\eta}\partial_{\xi}f & \partial_{\eta}\partial_{\eta}f & \partial_{\eta}\partial_{\zeta}f\\ \partial_{\zeta}\partial_{\xi}f & \partial_{\zeta}\partial_{\eta}f & \partial_{\zeta}\partial_{\zeta}f \end{bmatrix} \f$.
   * 
   * Since Hessian matrices are symmetric, only the upper part are stored.
   */
  using Hessian = algebra::Vector<Scalar, 6>;
  static Scalar &Get(Hessian &hessian, int row, int col) {
    if (row > col) { std::swap(row, col); }
    int i;
    switch (row) {
    case X: i = col; break;
    case Y: i = XZ + col; break;
    case Z: i = ZZ; break;
    default: assert(false);
    }
    return hessian[i];
  }
  static Scalar const &Get(Hessian const &hessian, int row, int col) {
    return Get(const_cast<Hessian &>(hessian), row, col);
  }

  virtual std::vector<Scalar> LocalToShapeFunctions(Scalar, Scalar, Scalar)
      const = 0;
  virtual std::vector<Local> LocalToShapeGradients(Scalar, Scalar, Scalar)
      const = 0;
  virtual std::vector<Hessian> LocalToShapeHessians(Local const &)
      const {
    return {};
  }

  std::vector<Scalar> LocalToShapeFunctions(const Local &xyz)
      const final {
    return LocalToShapeFunctions(xyz[X], xyz[Y], xyz[Z]);
  }
  std::vector<Local> LocalToShapeGradients(const Local &xyz)
      const final {
    return LocalToShapeGradients(xyz[X], xyz[Y], xyz[Z]);
  }

  Global LocalToGlobal(Scalar x_local, Scalar y_local, Scalar z_local)
      const {
    auto shapes = LocalToShapeFunctions(x_local, y_local, z_local);
    Global sum = this->GetGlobalCoord(0) * shapes[0];
    for (int i = 1, n = this->CountNodes(); i < n; ++i) {
      sum += this->GetGlobalCoord(i) * shapes[i];
    }
    return sum;
  }
  Global LocalToGlobal(const Local &xyz) const final {
    return LocalToGlobal(xyz[X], xyz[Y], xyz[Z]);
  }

  Jacobian LocalToJacobian(Scalar x_local, Scalar y_local, Scalar z_local)
      const {
    auto shapes = LocalToShapeGradients(x_local, y_local, z_local);
    Jacobian sum = this->GetGlobalCoord(0) * shapes[0].transpose();
    for (int i = 1, n = this->CountNodes(); i < n; ++i) {
      sum += this->GetGlobalCoord(i) * shapes[i].transpose();
    }
    return sum;
  }
  Jacobian LocalToJacobian(const Local &xyz) const final {
    return LocalToJacobian(xyz[X], xyz[Y], xyz[Z]);
  }
  /**
   * @brief \f$ \frac{\partial}{\partial\xi}\mathbf{J} = \frac{\partial}{\partial\xi}\begin{bmatrix} \partial_\xi x & \partial_\xi y & \partial_\xi z \\ \partial_\eta x & \partial_\eta y & \partial_\eta z \\ \partial_\zeta x & \partial_\zeta y & \partial_\zeta z \\ \end{bmatrix} \f$ in which \f$ \mathbf{J} = \begin{bmatrix} \partial_\xi x & \partial_\xi y & \partial_\xi z \\ \partial_\eta x & \partial_\eta y & \partial_\eta z \\ \partial_\zeta x & \partial_\zeta y & \partial_\zeta z \\ \end{bmatrix} \f$ is the transpose of `Element::Jacobian`.
   * 
   * @param xyz 
   * @return algebra::Vector<Jacobian, 3> 
   */
  algebra::Vector<Jacobian, 3> LocalToJacobianGradient(Local const &xyz)
      const {
    constexpr int YX = XY, ZY = YZ, ZX = XZ;
    auto hessians = LocalToShapeHessians(xyz);
    algebra::Vector<Jacobian, 3> grad;
    grad[X].setZero(); grad[Y].setZero(); grad[Z].setZero();
    for (int i = 0, n = this->CountNodes(); i < n; ++i) {
      auto &xyz = this->GetGlobalCoord(i);
      auto &hessian = hessians[i];
      grad[X].row(X) += xyz * hessian[XX];
      grad[X].row(Y) += xyz * hessian[XY];
      grad[X].row(Z) += xyz * hessian[XZ];
      grad[Y].row(X) += xyz * hessian[YX];
      grad[Y].row(Y) += xyz * hessian[YY];
      grad[Y].row(Z) += xyz * hessian[YZ];
      grad[Z].row(X) += xyz * hessian[ZX];
      grad[Z].row(Y) += xyz * hessian[ZY];
      grad[Z].row(Z) += xyz * hessian[ZZ];
    }
    return grad;
  }

  /**
   * @brief \f$ \frac{\partial J}{\partial \xi} = J\,\mathopen{\mathrm{tr}}\left(\mathbf{J}^{-1} \frac{\partial \mathbf{J}}{\partial \xi}\right) \f$, in which \f$ \mathbf{J} = \begin{bmatrix} \partial_\xi x & \partial_\xi y & \partial_\xi z \\ \partial_\eta x & \partial_\eta y & \partial_\eta z \\ \partial_\zeta x & \partial_\zeta y & \partial_\zeta z \\ \end{bmatrix} \f$ is the transpose of `Element::Jacobian` and \f$ \frac{\partial}{\partial \xi}\mathbf{J} \f$ is returned by `Cell::LocalToJacobianGradient`.
   * 
   * @param xyz 
   * @return algebra::Vector<Scalar, 3> 
   */
  algebra::Vector<Scalar, 3> LocalToJacobianDeterminantGradient(
      const Local &xyz) const {
    algebra::Vector<Scalar, 3> det_grad;
    auto mat_grad = LocalToJacobianGradient(xyz);
    Jacobian mat = LocalToJacobian(xyz).transpose();
    Scalar det = mat.determinant();
    Jacobian inv = mat.inverse();
    det_grad[X] = det * (inv * mat_grad[X]).trace();
    det_grad[Y] = det * (inv * mat_grad[Y]).trace();
    det_grad[Z] = det * (inv * mat_grad[Z]).trace();
    return det_grad;
  }

  Local GlobalToLocal(Scalar x_global, Scalar y_global, Scalar z_global,
      const Local &hint = Local(0, 0, 0)) const {
    Global xyz_global = {x_global, y_global, z_global};
    auto func = [this, &xyz_global](Local const &xyz_local) {
      auto res = LocalToGlobal(xyz_local);
      return res -= xyz_global;
    };
    auto jac = [this](Local const &xyz_local) {
      return LocalToJacobian(xyz_local);
    };
    return root(func, hint, jac);
  }
  Local GlobalToLocal(const Global &xyz, const Local &hint = Local(0, 0, 0)) const {
    return GlobalToLocal(xyz[X], xyz[Y], xyz[Z], hint);
  }

  /**
   * @brief Sort `cell_nodes` by `face_nodes`, so that the right-hand normal of the Face point out from the Cell.
   * 
   * @param cell_nodes  The node id list of a Cell.
   * @param face_nodes  The node id list of a Face.
   * @param face_n_node  Number of nodes on the Face.
   */
  virtual void SortNodesOnFace(const size_t *cell_nodes, size_t *face_nodes,
      int face_n_node) const = 0;

 private:
  template <typename Callable, typename MatJ>
  static Global root(
      Callable &&func, Global x, MatJ &&matj, Scalar xtol = 1e-5) {
    Global res;
    do {
      res = matj(x).partialPivLu().solve(func(x));
      x -= res;
    } while (res.norm() > xtol);
    return x;
  }
};

/**
 * @brief A generic wrapper of the virtual SortNodesOnFace method.
 * 
 * @tparam Scalar  Same as Cell::Scalar.
 * @tparam T  Type of integers in the 1st list.
 * @tparam U  Type of integers in the 2nd list.
 * @param cell  The Cell holding the Face.
 * @param cell_nodes  The node id list of the Cell.
 * @param face_nodes  The node id list of the Face.
 * @param face_n_node  Number of nodes on the Face.
 */
template<std::floating_point Scalar, std::integral T, std::integral U>
void SortNodesOnFace(const Cell<Scalar> &cell, const T *cell_nodes,
    U *face_nodes, int face_n_node) {
  size_t cell_nodes_copy[64], face_nodes_copy[32];
  size_t *cell_node_list, *face_node_list;
  if (sizeof(T) == sizeof(size_t)) {
    cell_node_list = reinterpret_cast<size_t *>(const_cast<T *>(cell_nodes));
  } else {
    cell_node_list = cell_nodes_copy;
    auto n_nodes = cell.CountNodes();
    std::copy_n(cell_nodes, n_nodes, cell_node_list);
  }
  if (sizeof(U) == sizeof(size_t)) {
    face_node_list = reinterpret_cast<size_t *>(const_cast<U *>(face_nodes));
  } else {
    face_node_list = face_nodes_copy;
    std::copy_n(face_nodes, face_n_node, face_node_list);
  }
  // Delegate the real work or sorting to the virtual function.
  cell.SortNodesOnFace(cell_node_list, face_node_list, face_n_node);
  if (sizeof(U) != sizeof(size_t)) {
    std::copy_n(face_node_list, face_n_node, face_nodes);
  }
}


}  // namespace geometry
}  // namespace mini

#endif  // MINI_GEOMETRY_CELL_HPP_
