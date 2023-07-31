//  Copyright 2023 PEI Weicheng
#ifndef MINI_LAGRANGE_WEDGE_HPP_
#define MINI_LAGRANGE_WEDGE_HPP_

#include <concepts>

#include <cassert>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <type_traits>
#include <utility>
#include <vector>

#include "mini/lagrange/cell.hpp"

namespace mini {
namespace lagrange {

/**
 * @brief Abstract coordinate map on wedge elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Wedge : public Cell<Scalar> {
  using Base = Cell<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  int CountCorners() const final {
    return 6;
  }
  const Global &center() const final {
    return center_;
  }

 protected:
  Global center_;
  void BuildCenter() {
    Scalar a = 1.0 / 3;
    center_ = this->LocalToGlobal(a, a, 0);
  }
};

/**
 * @brief Coordinate map on 6-node wedge elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Wedge6 : public Wedge<Scalar> {
  using Base = Wedge<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  static constexpr int kNodes = 6;

 private:
  std::array<Global, kNodes> global_coords_;
  static const std::array<Local, kNodes> local_coords_;
  static const std::array<std::array<int, 3>, 2> triangles_;
  static const std::array<std::array<int, 4>, 3> quadrangles_;

 public:
  int CountNodes() const override {
    return kNodes;
  }
  void SortNodesOnFace(const size_t *cell_nodes, size_t *face_nodes,
      int face_n_node) const final {
    if (face_n_node == 3) {
      int i_face = 0;
      for (int f = 0; f < 3; ++f) {
        if (cell_nodes[4] == face_nodes[f]) {
          i_face = 1;
          break;
        }
      }
      for (int i = 0; i < 3; ++i) {
        face_nodes[i] = cell_nodes[triangles_[i_face][i]];
      }
    } else {
      assert(face_n_node == 4);
      int c_sum = 0;  // sum of c's in (0, 1, 2)
      for (int f = 0; f < 4; ++f) {
        for (int c = 0; c < 3; ++c) {
          if (cell_nodes[c] == face_nodes[f]) {
            c_sum += c;
            break;
          }
        }
      }
      int i_face;
      switch (c_sum) {
      case 1: i_face = 0; break;
      case 2: i_face = 2; break;
      case 3: i_face = 1; break;
      default: assert(false);
      }
      for (int i = 0; i < 4; ++i) {
        face_nodes[i] = cell_nodes[quadrangles_[i_face][i]];
      }
    }
  }

 public:
  std::vector<Scalar> LocalToShapeFunctions(
      Scalar a_local, Scalar b_local, Scalar z_local) const override {
    auto shapes = std::vector<Scalar>(kNodes);
    auto c_local = 1.0 - a_local - b_local;
    auto factor_z = (1 - z_local) / 2;
    shapes[0] = a_local * factor_z;
    shapes[1] = b_local * factor_z;
    shapes[2] = c_local * factor_z;
    factor_z += z_local;
    shapes[3] = a_local * factor_z;
    shapes[4] = b_local * factor_z;
    shapes[5] = c_local * factor_z;
    return shapes;
  }
  std::vector<Local> LocalToShapeGradients(
      Scalar a_local, Scalar b_local, Scalar z_local) const override {
    auto grads = std::vector<Local>(kNodes);
    constexpr int A{0}, B{1};
    auto c_local = 1.0 - a_local - b_local;
    auto factor_z = (1 - z_local) / 2;
    // shapes[0] = a_local * factor_z
    grads[0][A] = factor_z;
    grads[0][B] = 0;
    grads[0][Z] = -0.5 * a_local;
    // shapes[1] = b_local * factor_z
    grads[1][A] = 0;
    grads[1][B] = factor_z;
    grads[1][Z] = -0.5 * b_local;
    // shapes[2] = (1 - a_local - b_local) * factor_z
    grads[2][A] = -factor_z;
    grads[2][B] = -factor_z;
    grads[2][Z] = -0.5 * c_local;
    factor_z += z_local;
    // shapes[3] = a_local * factor_z;
    grads[3][A] = factor_z;
    grads[3][B] = 0;
    grads[3][Z] = +0.5 * a_local;
    // shapes[4] = b_local * factor_z;
    grads[4][A] = 0;
    grads[4][B] = factor_z;
    grads[4][Z] = +0.5 * b_local;
    // shapes[5] = c_local * factor_z;
    grads[5][A] = -factor_z;
    grads[5][B] = -factor_z;
    grads[5][Z] = +0.5 * c_local;
    return grads;
  }

 public:
  Global const &GetGlobalCoord(int i) const override {
    assert(0 <= i && i < CountNodes());
    return global_coords_[i];
  }
  Local const &GetLocalCoord(int i) const override {
    assert(0 <= i && i < CountNodes());
    return local_coords_[i];
  }

 public:
  Wedge6(
      Global const &p0, Global const &p1, Global const &p2, Global const &p3,
      Global const &p4, Global const &p5) {
    global_coords_[0] = p0; global_coords_[1] = p1;
    global_coords_[2] = p2; global_coords_[3] = p3;
    global_coords_[4] = p4; global_coords_[5] = p5;
    this->BuildCenter();
  }
  Wedge6(std::initializer_list<Global> il) {
    assert(il.size() == kNodes);
    auto p = il.begin();
    for (int i = 0; i < kNodes; ++i) {
      global_coords_[i] = p[i];
    }
    this->BuildCenter();
  }
};
// initialization of static const members:
template <std::floating_point Scalar>
const std::array<typename Wedge6<Scalar>::Local, 6>
Wedge6<Scalar>::local_coords_{
  Wedge6::Local(1, 0, -1), Wedge6::Local(0, 1, -1), Wedge6::Local(0, 0, -1),
  Wedge6::Local(1, 0, +1), Wedge6::Local(0, 1, +1), Wedge6::Local(0, 0, +1)
};

/* See http://cgns.github.io/CGNS_docs_current/sids/conv.figs/penta_6.png for node numbering.
*/
template <std::floating_point Scalar>
const std::array<std::array<int, 3>, 2>
Wedge6<Scalar>::triangles_{
  0, 2, 1,
  3, 4, 5,
};
template <std::floating_point Scalar>
const std::array<std::array<int, 4>, 3>
Wedge6<Scalar>::quadrangles_{
  0, 1, 4, 3,
  1, 2, 5, 4,
  0, 3, 5, 2,
};

}  // namespace lagrange
}  // namespace mini

#endif  // MINI_LAGRANGE_WEDGE_HPP_
