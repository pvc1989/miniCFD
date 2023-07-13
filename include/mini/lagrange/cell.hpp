//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_LAGRANGE_CELL_HPP_
#define MINI_LAGRANGE_CELL_HPP_

#include <algorithm>
#include <concepts>

#include "mini/algebra/eigen.hpp"

namespace mini {
namespace lagrange {

/**
 * @brief Abstract coordinate map on volume elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Cell {
  using Mat3x3 = algebra::Matrix<Scalar, 3, 3>;
  using Mat3x1 = algebra::Matrix<Scalar, 3, 1>;

 public:
  using Real = Scalar;
  using LocalCoord = Mat3x1;
  using GlobalCoord = Mat3x1;
  using Jacobian = Mat3x3;

  virtual ~Cell() noexcept = default;
  virtual std::vector<Scalar> LocalToShapeFunctions(Scalar, Scalar, Scalar) const = 0;
  virtual std::vector<LocalCoord> LocalToShapeGradients(Scalar, Scalar, Scalar) const = 0;
  virtual int CountVertices() const = 0;
  virtual int CountNodes() const = 0;
  virtual const LocalCoord &GetLocalCoord(int i) const = 0;
  virtual const GlobalCoord &GetGlobalCoord(int i) const = 0;
  virtual const GlobalCoord &center() const = 0;

  /**
   * @brief Sort `cell_nodes` by `face_nodes`, so that the right-hand normal of the Face point out from the Cell.
   * 
   * @param cell_nodes  The node id list of a Cell.
   * @param face_nodes  The node id list of a Face.
   */
  virtual void SortNodesOnFace(const size_t *cell_nodes, size_t *face_nodes) const = 0;

  static constexpr int CellDim() {
    return 3;
  }
  static constexpr int PhysDim() {
    return 3;
  }

  GlobalCoord LocalToGlobal(Scalar x_local, Scalar y_local, Scalar z_local)
      const {
    auto shapes = LocalToShapeFunctions(x_local, y_local, z_local);
    GlobalCoord sum = GetGlobalCoord(0) * shapes[0];
    for (int i = 1; i < CountNodes(); ++i) {
      sum += GetGlobalCoord(i) * shapes[i];
    }
    return sum;
  }
  GlobalCoord LocalToGlobal(const LocalCoord &xyz) const {
    return LocalToGlobal(xyz[0], xyz[1], xyz[2]);
  }

  Jacobian LocalToJacobian(Scalar x_local, Scalar y_local, Scalar z_local)
      const {
    auto shapes = LocalToShapeGradients(x_local, y_local, z_local);
    Jacobian sum = GetGlobalCoord(0) * shapes[0].transpose();
    for (int i = 1; i < CountNodes(); ++i) {
      sum += GetGlobalCoord(i) * shapes[i].transpose();
    }
    return sum;
  }
  Jacobian LocalToJacobian(const LocalCoord &xyz) const {
    return LocalToJacobian(xyz[0], xyz[1], xyz[2]);
  }

  LocalCoord GlobalToLocal(Scalar x_global, Scalar y_global, Scalar z_global)
      const {
    Mat3x1 xyz_global = {x_global, y_global, z_global};
    auto func = [this, &xyz_global](Mat3x1 const &xyz_local) {
      auto res = LocalToGlobal(xyz_local);
      return res -= xyz_global;
    };
    auto jac = [this](LocalCoord const &xyz_local) {
      return LocalToJacobian(xyz_local);
    };
    Mat3x1 xyz0 = {0, 0, 0};
    return root(func, xyz0, jac);
  }
  LocalCoord GlobalToLocal(const GlobalCoord &xyz) const {
    return GlobalToLocal(xyz[0], xyz[1], xyz[2]);
  }

 private:
  template <typename Callable, typename MatJ>
  static Mat3x1 root(
      Callable &&func, Mat3x1 x, MatJ &&matj, Scalar xtol = 1e-5) {
    Mat3x1 res;
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
  cell.SortNodesOnFace(cell_node_list, face_node_list);
  if (sizeof(U) != sizeof(size_t)) {
    std::copy_n(face_node_list, face_n_node, face_nodes);
  }
}


}  // namespace lagrange
}  // namespace mini

#endif  // MINI_LAGRANGE_CELL_HPP_
