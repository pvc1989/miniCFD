//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_LAGRANGE_CELL_HPP_
#define MINI_LAGRANGE_CELL_HPP_

#include <concepts>

#include <algorithm>
#include <vector>

#include "mini/lagrange/element.hpp"

namespace mini {
namespace lagrange {

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

  virtual std::vector<Scalar> LocalToShapeFunctions(Scalar, Scalar, Scalar)
      const = 0;
  virtual std::vector<Local> LocalToShapeGradients(Scalar, Scalar, Scalar)
      const = 0;

  std::vector<Scalar> LocalToShapeFunctions(const Local &xyz)
      const override {
    return LocalToShapeFunctions(xyz[X], xyz[Y], xyz[Z]);
  }
  std::vector<Local> LocalToShapeGradients(const Local &xyz)
      const override {
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
  Global LocalToGlobal(const Local &xyz) const override {
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
  Jacobian LocalToJacobian(const Local &xyz) const override {
    return LocalToJacobian(xyz[X], xyz[Y], xyz[Z]);
  }

  Local GlobalToLocal(Scalar x_global, Scalar y_global, Scalar z_global)
      const {
    Global xyz_global = {x_global, y_global, z_global};
    auto func = [this, &xyz_global](Local const &xyz_local) {
      auto res = LocalToGlobal(xyz_local);
      return res -= xyz_global;
    };
    auto jac = [this](Local const &xyz_local) {
      return LocalToJacobian(xyz_local);
    };
    Global xyz0 = {0, 0, 0};
    return root(func, xyz0, jac);
  }
  Local GlobalToLocal(const Global &xyz) const {
    return GlobalToLocal(xyz[X], xyz[Y], xyz[Z]);
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
    cell_node_list = (size_t *)(cell_nodes);
  } else {
    cell_node_list = cell_nodes_copy;
    auto n_nodes = cell.CountNodes();
    std::copy_n(cell_nodes, n_nodes, cell_node_list);
  }
  if (sizeof(U) == sizeof(size_t)) {
    face_node_list = (size_t *)(face_nodes);
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


}  // namespace lagrange
}  // namespace mini

#endif  // MINI_LAGRANGE_CELL_HPP_
