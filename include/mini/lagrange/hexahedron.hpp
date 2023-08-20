//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_LAGRANGE_HEXAHEDRON_HPP_
#define MINI_LAGRANGE_HEXAHEDRON_HPP_

#include <concepts>

#include <cassert>

#include <array>
#include <initializer_list>
#include <vector>

#include "mini/lagrange/element.hpp"
#include "mini/lagrange/cell.hpp"

namespace mini {
namespace lagrange {

/**
 * @brief Abstract coordinate map on hexahedral elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Hexahedron : public Cell<Scalar> {
  using Base = Cell<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  int CountCorners() const final {
    return 8;
  }
  const Global &center() const final {
    return center_;
  }

 protected:
  Global center_;
  void _BuildCenter() final {
    Scalar a = 0;
    center_ = this->LocalToGlobal(a, a, a);
  }
};

/**
 * @brief Coordinate map on 8-node hexahedral elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Hexahedron8 : public Hexahedron<Scalar> {
  using Base = Hexahedron<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  static constexpr int kNodes = 8;
  static constexpr int kFaces = 6;

 private:
  std::array<Global, kNodes> global_coords_;
  static const std::array<Local, kNodes> local_coords_;
  static const std::array<std::array<int, 4>, kFaces> faces_;

 public:
  int CountNodes() const final {
    return kNodes;
  }
  void SortNodesOnFace(const size_t *cell_nodes, size_t *face_nodes,
      int face_n_node) const final {
    assert(4 == face_n_node);
    int cnt = 0, nid = 0, sum = 0;
    while (cnt < 3) {
      auto curr_node = cell_nodes[nid];
      for (int i = 0; i < 4; ++i) {
        if (face_nodes[i] == curr_node) {
          sum += nid;
          ++cnt;
          break;
        }
      }
      ++nid;
    }
    int i_face;
    switch (sum) {
    case 3:
      i_face = 0; break;
    case 5:
      i_face = 1; break;
    case 7:
      i_face = 4; break;
    case 8:
      i_face = 2; break;
    case 11:
      i_face = 3; break;
    case 15:
      i_face = 5; break;
    default:
      assert(false);
    }
    for (int i = 0; i < 4; ++i) {
      face_nodes[i] = cell_nodes[faces_[i_face][i]];
    }
  }

 public:
  static void LocalToShapeFunctions(Scalar x_local, Scalar y_local,
      Scalar z_local, Scalar *shapes) {
    for (int i = 0; i < kNodes; ++i) {
      auto &local_i = local_coords_[i];
      auto val = (1 + local_i[X] * x_local);
          val *= (1 + local_i[Y] * y_local);
          val *= (1 + local_i[Z] * z_local);
      shapes[i] = val / 8;
    }
  }
  std::vector<Scalar> LocalToShapeFunctions(
      Scalar x_local, Scalar y_local, Scalar z_local) const final {
    auto shapes = std::vector<Scalar>(kNodes);
    LocalToShapeFunctions(x_local, y_local, z_local, shapes.data());
    return shapes;
  }
  static void LocalToShapeGradients(Scalar x_local, Scalar y_local,
      Scalar z_local, Local *grads) {
    std::array<Scalar, kNodes> factor_xy, factor_yz, factor_zx;
    for (int i = 0; i < kNodes; ++i) {
      auto &local_i = local_coords_[i];
      auto factor_x = (1 + local_i[X] * x_local);
      auto factor_y = (1 + local_i[Y] * y_local);
      auto factor_z = (1 + local_i[Z] * z_local);
      factor_xy[i] = factor_x * factor_y / 8;
      factor_yz[i] = factor_y * factor_z / 8;
      factor_zx[i] = factor_z * factor_x / 8;
    }
    for (int i = 0; i < kNodes; ++i) {
      auto &local_i = local_coords_[i];
      auto &grad_i = grads[i];
      grad_i[X] = local_i[X] * factor_yz[i];
      grad_i[Y] = local_i[Y] * factor_zx[i];
      grad_i[Z] = local_i[Z] * factor_xy[i];
    }
  }
  std::vector<Local> LocalToShapeGradients(
      Scalar x_local, Scalar y_local, Scalar z_local) const final {
    auto grads = std::vector<Local>(kNodes);
    LocalToShapeGradients(x_local, y_local, z_local, grads.data());
    return grads;
  }

 public:
  Global const &GetGlobalCoord(int i) const final {
    assert(0 <= i && i < CountNodes());
    return global_coords_[i];
  }
  Local const &GetLocalCoord(int i) const final {
    assert(0 <= i && i < CountNodes());
    return local_coords_[i];
  }

 public:
  Hexahedron8(
      Global const &p0, Global const &p1,
      Global const &p2, Global const &p3,
      Global const &p4, Global const &p5,
      Global const &p6, Global const &p7) {
    global_coords_[0] = p0; global_coords_[1] = p1;
    global_coords_[2] = p2; global_coords_[3] = p3;
    global_coords_[4] = p4; global_coords_[5] = p5;
    global_coords_[6] = p6; global_coords_[7] = p7;
    this->_BuildCenter();
  }

  friend void lagrange::_Build(Hexahedron8 *,
      std::initializer_list<Global>);
  Hexahedron8(std::initializer_list<Global> il) {
    lagrange::_Build(this, il);
  }
};
// initialization of static const members:
template <std::floating_point Scalar>
const std::array<typename Hexahedron8<Scalar>::Local, 8>
Hexahedron8<Scalar>::local_coords_{
  Hexahedron8::Local(-1, -1, -1), Hexahedron8::Local(+1, -1, -1),
  Hexahedron8::Local(+1, +1, -1), Hexahedron8::Local(-1, +1, -1),
  Hexahedron8::Local(-1, -1, +1), Hexahedron8::Local(+1, -1, +1),
  Hexahedron8::Local(+1, +1, +1), Hexahedron8::Local(-1, +1, +1)
};

template <std::floating_point Scalar>
const std::array<std::array<int, 4>, 6>
Hexahedron8<Scalar>::faces_{
  /* See http://cgns.github.io/CGNS_docs_current/sids/conv.figs/hexa_8.png for node numbering.
   * Faces can be distinguished by the sum of the three minimum node ids.
   */
  0, 3, 2, 1/* 0 + 2 + 1 == 3 */, 0, 1, 5, 4/* 0 + 1 + 4 == 5 */,
  1, 2, 6, 5/* 1 + 2 + 5 == 8 */, 2, 3, 7, 6/* 2 + 3 + 6 == 11 */,
  0, 4, 7, 3/* 0 + 4 + 4 == 7 */, 4, 5, 6, 7/* 4 + 5 + 6 == 15 */
};

/**
 * @brief Coordinate map on 20-node hexahedral elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Hexahedron20 : public Hexahedron<Scalar> {
  using Base = Hexahedron<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  static constexpr int kNodes = 20;
  static constexpr int kFaces = 6;

 private:
  std::array<Global, kNodes> global_coords_;
  static const std::array<Local, kNodes> local_coords_;
  static const std::array<std::array<int, 8>, kFaces> faces_;

 public:
  int CountNodes() const final {
    return kNodes;
  }
  void SortNodesOnFace(const size_t *cell_nodes, size_t *face_nodes,
      int face_n_node) const final {
    assert(8 == face_n_node);
    int cnt = 0, c = 0, c_sum = 0;
    while (cnt < 3) {
      auto curr_node = cell_nodes[c];
      for (int f = 0; f < 8; ++f) {
        if (face_nodes[f] == curr_node) {
          c_sum += c;
          ++cnt;
          break;
        }
      }
      ++c;
    }
    int i_face;
    switch (c_sum) {
    case 3:
      i_face = 0; break;
    case 5:
      i_face = 1; break;
    case 7:
      i_face = 4; break;
    case 8:
      i_face = 2; break;
    case 11:
      i_face = 3; break;
    case 15:
      i_face = 5; break;
    default:
      assert(false);
    }
    for (int i = 0; i < 8; ++i) {
      face_nodes[i] = cell_nodes[faces_[i_face][i]];
    }
  }

 private:
  static void LocalToNewShapeFunctions(Scalar x_local, Scalar y_local,
      Scalar z_local, Scalar *shapes) {
    // x_local = 0
    Scalar factor_x = (1 - x_local * x_local) / 4;
    for (int a : {8, 10, 16, 18}) {
      shapes[a] = factor_x
          * (1 + local_coords_[a][Y] * y_local)
          * (1 + local_coords_[a][Z] * z_local);
    }
    // y_local = 0
    Scalar factor_y = (1 - y_local * y_local) / 4;
    for (int a : {9, 11, 17, 19}) {
      shapes[a] = factor_y
          * (1 + local_coords_[a][X] * x_local)
          * (1 + local_coords_[a][Z] * z_local);
    }
    // z_local = 0
    Scalar factor_z = (1 - z_local * z_local) / 4;
    for (int a : {12, 13, 14, 15}) {
      shapes[a] = factor_z
          * (1 + local_coords_[a][X] * x_local)
          * (1 + local_coords_[a][Y] * y_local);
    }
  }
  static void LocalToNewShapeGradients(Scalar x_local, Scalar y_local,
      Scalar z_local, Local *grads) {
    // x_local = 0
    Scalar shape_x = (1 - x_local * x_local) / 4;
    Scalar grad_x = -x_local / 2;
    for (int a : {8, 10, 16, 18}) {
      Scalar shape_y = (1 + local_coords_[a][Y] * y_local);
      Scalar shape_z = (1 + local_coords_[a][Z] * z_local);
      grads[a][X] = grad_x * shape_y * shape_z;
      grads[a][Y] = shape_x * local_coords_[a][Y] * shape_z;
      grads[a][Z] = shape_x * shape_y * local_coords_[a][Z];
    }
    // y_local = 0
    Scalar shape_y = (1 - y_local * y_local) / 4;
    Scalar grad_y = -y_local / 2;
    for (int a : {9, 11, 17, 19}) {
      Scalar shape_x = (1 + local_coords_[a][X] * x_local);
      Scalar shape_z = (1 + local_coords_[a][Z] * z_local);
      grads[a][X] = local_coords_[a][X] * shape_y * shape_z;
      grads[a][Y] = shape_x * grad_y * shape_z;
      grads[a][Z] = shape_x * shape_y * local_coords_[a][Z];
    }
    // z_local = 0
    Scalar shape_z = (1 - z_local * z_local) / 4;
    Scalar grad_z = -z_local / 2;
    for (int a : {12, 13, 14, 15}) {
      Scalar shape_x = (1 + local_coords_[a][X] * x_local);
      Scalar shape_y = (1 + local_coords_[a][Y] * y_local);
      grads[a][X] = local_coords_[a][X] * shape_y * shape_z;
      grads[a][Y] = shape_x * local_coords_[a][Y] * shape_z;
      grads[a][Z] = shape_x * shape_y * grad_z;
    }
  }

 public:
  static void LocalToShapeFunctions(Scalar x_local, Scalar y_local,
      Scalar z_local, Scalar *shapes) {
    Hexahedron8<Scalar>::LocalToShapeFunctions(
        x_local, y_local, z_local, shapes);
    LocalToNewShapeFunctions(x_local, y_local, z_local, shapes);
    for (int b = 8; b < 20; ++b) {
      Scalar x_b = local_coords_[b][X];
      Scalar y_b = local_coords_[b][Y];
      Scalar z_b = local_coords_[b][Z];
      Scalar old_shapes_on_new_nodes[8];
      Hexahedron8<Scalar>::LocalToShapeFunctions(
          x_b, y_b, z_b, old_shapes_on_new_nodes);
      for (int a = 0; a < 8; ++a) {
        shapes[a] -= old_shapes_on_new_nodes[a] * shapes[b];
      }
    }
  }
  std::vector<Scalar> LocalToShapeFunctions(
      Scalar x_local, Scalar y_local, Scalar z_local) const final {
    auto shapes = std::vector<Scalar>(kNodes);
    LocalToShapeFunctions(x_local, y_local, z_local, shapes.data());
    return shapes;
  }
  static void LocalToShapeGradients(Scalar x_local, Scalar y_local,
      Scalar z_local, Local *grads) {
    Hexahedron8<Scalar>::LocalToShapeGradients(
        x_local, y_local, z_local, grads);
    LocalToNewShapeGradients(x_local, y_local, z_local, grads);
    for (int b = 8; b < 20; ++b) {
      Scalar x_b = local_coords_[b][X];
      Scalar y_b = local_coords_[b][Y];
      Scalar z_b = local_coords_[b][Z];
      Scalar old_shapes_on_new_nodes[8];
      Hexahedron8<Scalar>::LocalToShapeFunctions(
          x_b, y_b, z_b, old_shapes_on_new_nodes);
      for (int a = 0; a < 8; ++a) {
        grads[a] -= old_shapes_on_new_nodes[a] * grads[b];
      }
    }
  }
  std::vector<Local> LocalToShapeGradients(
      Scalar x_local, Scalar y_local, Scalar z_local) const final {
    auto grads = std::vector<Local>(kNodes);
    LocalToShapeGradients(x_local, y_local, z_local, grads.data());
    return grads;
  }

 public:
  Global const &GetGlobalCoord(int i) const final {
    assert(0 <= i && i < CountNodes());
    return global_coords_[i];
  }
  Local const &GetLocalCoord(int i) const final {
    assert(0 <= i && i < CountNodes());
    return local_coords_[i];
  }

 public:
  Hexahedron20(
      Global const &p0, Global const &p1, Global const &p2, Global const &p3,
      Global const &p4, Global const &p5, Global const &p6, Global const &p7,
      Global const &p8, Global const &p9, Global const &p10, Global const &p11,
      Global const &p12, Global const &p13,
      Global const &p14, Global const &p15,
      Global const &p16, Global const &p17,
      Global const &p18, Global const &p19) {
    global_coords_[0] = p0; global_coords_[1] = p1;
    global_coords_[2] = p2; global_coords_[3] = p3;
    global_coords_[4] = p4; global_coords_[5] = p5;
    global_coords_[6] = p6; global_coords_[7] = p7;
    global_coords_[8] = p8; global_coords_[9] = p9;
    global_coords_[10] = p10; global_coords_[11] = p11;
    global_coords_[12] = p12; global_coords_[13] = p13;
    global_coords_[14] = p14; global_coords_[15] = p15;
    global_coords_[16] = p16; global_coords_[17] = p17;
    global_coords_[18] = p19; global_coords_[19] = p19;
    this->_BuildCenter();
  }

  friend void lagrange::_Build(Hexahedron20 *,
      std::initializer_list<Global>);
  Hexahedron20(std::initializer_list<Global> il) {
    lagrange::_Build(this, il);
  }
};
// initialization of static const members:
template <std::floating_point Scalar>
const std::array<typename Hexahedron20<Scalar>::Local, 20>
Hexahedron20<Scalar>::local_coords_{
  // corner nodes on the bottom face
  Hexahedron20::Local(-1, -1, -1), Hexahedron20::Local(+1, -1, -1),
  Hexahedron20::Local(+1, +1, -1), Hexahedron20::Local(-1, +1, -1),
  // corner nodes on the top face
  Hexahedron20::Local(-1, -1, +1), Hexahedron20::Local(+1, -1, +1),
  Hexahedron20::Local(+1, +1, +1), Hexahedron20::Local(-1, +1, +1),
  // mid-edge nodes on the bottom face
  Hexahedron20::Local(0, -1, -1), Hexahedron20::Local(+1, 0, -1),
  Hexahedron20::Local(0, +1, -1), Hexahedron20::Local(-1, 0, -1),
  // mid-edge nodes on vertical edges
  Hexahedron20::Local(-1, -1, 0), Hexahedron20::Local(+1, -1, 0),
  Hexahedron20::Local(+1, +1, 0), Hexahedron20::Local(-1, +1, 0),
  // mid-edge nodes on the top face
  Hexahedron20::Local(0, -1, +1), Hexahedron20::Local(+1, 0, +1),
  Hexahedron20::Local(0, +1, +1), Hexahedron20::Local(-1, 0, +1),
};

template <std::floating_point Scalar>
const std::array<std::array<int, 8>, 6>
Hexahedron20<Scalar>::faces_{
  /* See http://cgns.github.io/CGNS_docs_current/sids/conv.figs/hexa_20.png for node numbering.
   * Faces can be distinguished by the sum of the three minimum node ids.
   */
  /* zeta = -1 */0, 3, 2, 1, 11, 10, 9, 8,
  /*  eta = -1 */0, 1, 5, 4, 8, 13, 16, 12,
  /*   xi = +1 */1, 2, 6, 5, 9, 14, 17, 13,
  /*  eta = +1 */2, 3, 7, 6, 10, 15, 18, 14,
  /*   xi = -1 */0, 4, 7, 3, 12, 19, 15, 11,
  /* zeta = +1 */4, 5, 6, 7, 16, 17, 18, 19,
};

}  // namespace lagrange
}  // namespace mini

#endif  // MINI_LAGRANGE_HEXAHEDRON_HPP_
