//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_LAGRANGE_HEXAHEDRON_HPP_
#define MINI_LAGRANGE_HEXAHEDRON_HPP_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstring>
#include <type_traits>
#include <utility>

#include "mini/algebra/eigen.hpp"
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
  using typename Base::LocalCoord;
  using typename Base::GlobalCoord;
  using typename Base::Jacobian;

  int CountCorners() const override final {
    return 8;
  }
  const GlobalCoord &center() const override final {
    return center_;
  }

 protected:
  GlobalCoord center_;
  void BuildCenter() {
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
  using typename Base::LocalCoord;
  using typename Base::GlobalCoord;
  using typename Base::Jacobian;

  static constexpr int kNodes = 8;
  static constexpr int kFaces = 6;

 private:
  std::array<GlobalCoord, kNodes> global_coords_;
  static const std::array<LocalCoord, kNodes> local_coords_;
  static const std::array<std::array<int, 4>, kFaces> faces_;

 public:
  int CountNodes() const override {
    return kNodes;
  }
  void SortNodesOnFace(const size_t *cell_nodes, size_t *face_nodes) const override {
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
  std::vector<Scalar> LocalToShapeFunctions(
      Scalar x_local, Scalar y_local, Scalar z_local) const override {
    auto shapes = std::vector<Scalar>(kNodes);
    for (int i = 0; i < kNodes; ++i) {
      auto &local_i = GetLocalCoord(i);
      auto val = (1 + local_i[X] * x_local);
          val *= (1 + local_i[Y] * y_local);
          val *= (1 + local_i[Z] * z_local);
      shapes[i] = val / 8;
    }
    return shapes;
  }
  std::vector<LocalCoord> LocalToShapeGradients(
      Scalar x_local, Scalar y_local, Scalar z_local) const override {
    auto shapes = std::vector<LocalCoord>(kNodes);
    std::array<Scalar, kNodes> factor_xy, factor_yz, factor_zx;
    for (int i = 0; i < kNodes; ++i) {
      auto &local_i = GetLocalCoord(i);
      auto factor_x = (1 + local_i[X] * x_local);
      auto factor_y = (1 + local_i[Y] * y_local);
      auto factor_z = (1 + local_i[Z] * z_local);
      factor_xy[i] = factor_x * factor_y / 8;
      factor_yz[i] = factor_y * factor_z / 8;
      factor_zx[i] = factor_z * factor_x / 8;
    }
    for (int i = 0; i < kNodes; ++i) {
      auto &local_i = GetLocalCoord(i);
      auto &shape_i = shapes[i];
      shape_i[X] = local_i[X] * factor_yz[i];
      shape_i[Y] = local_i[Y] * factor_zx[i];
      shape_i[Z] = local_i[Z] * factor_xy[i];
    }
    return shapes;
  }

 public:
  GlobalCoord const &GetGlobalCoord(int q) const override {
    return global_coords_[q];
  }
  LocalCoord const &GetLocalCoord(int q) const override {
    return local_coords_[q];
  }

 public:
  Hexahedron8(
      GlobalCoord const &p0, GlobalCoord const &p1,
      GlobalCoord const &p2, GlobalCoord const &p3,
      GlobalCoord const &p4, GlobalCoord const &p5,
      GlobalCoord const &p6, GlobalCoord const &p7) {
    global_coords_[0] = p0; global_coords_[1] = p1;
    global_coords_[2] = p2; global_coords_[3] = p3;
    global_coords_[4] = p4; global_coords_[5] = p5;
    global_coords_[6] = p6; global_coords_[7] = p7;
    this->BuildCenter();
  }
  Hexahedron8(std::initializer_list<GlobalCoord> il) {
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
const std::array<typename Hexahedron8<Scalar>::LocalCoord, 8>
Hexahedron8<Scalar>::local_coords_{
  Hexahedron8::LocalCoord(-1, -1, -1), Hexahedron8::LocalCoord(+1, -1, -1),
  Hexahedron8::LocalCoord(+1, +1, -1), Hexahedron8::LocalCoord(-1, +1, -1),
  Hexahedron8::LocalCoord(-1, -1, +1), Hexahedron8::LocalCoord(+1, -1, +1),
  Hexahedron8::LocalCoord(+1, +1, +1), Hexahedron8::LocalCoord(-1, +1, +1)
};

template <std::floating_point Scalar>
const std::array<std::array<int, 4>, 6>
Hexahedron8<Scalar>::faces_{
  // See http://cgns.github.io/CGNS_docs_current/sids/conv.figs/hexa_8.png for node numbering.
  // Faces can be distinguished by the sum of the three minimum node ids.
  0, 3, 2, 1/* 0 + 2 + 1 == 3 */, 0, 1, 5, 4/* 0 + 1 + 4 == 5 */,
  1, 2, 6, 5/* 1 + 2 + 5 == 8 */, 2, 3, 7, 6/* 2 + 3 + 6 == 11 */,
  0, 4, 7, 3/* 0 + 4 + 4 == 7 */, 4, 5, 6, 7/* 4 + 5 + 6 == 15 */
};

}  // namespace lagrange
}  // namespace mini

#endif  // MINI_LAGRANGE_HEXAHEDRON_HPP_
