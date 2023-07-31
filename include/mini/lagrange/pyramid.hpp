//  Copyright 2023 PEI Weicheng
#ifndef MINI_LAGRANGE_PYRAMID_HPP_
#define MINI_LAGRANGE_PYRAMID_HPP_

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
 * @brief Abstract coordinate map on pyramidal elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Pyramid : public Cell<Scalar> {
  using Base = Cell<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  int CountCorners() const final {
    return 5;
  }
  const Global &center() const final {
    return center_;
  }

 protected:
  Global center_;
  void BuildCenter() {
    Scalar a = 0;
    center_ = this->LocalToGlobal(a, a, a);
  }
};

/**
 * @brief Coordinate map on 5-node pyramidal elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Pyramid5 : public Pyramid<Scalar> {
  using Base = Pyramid<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  static constexpr int kNodes = 5;

 private:
  std::array<Global, kNodes> global_coords_;
  static const std::array<Local, kNodes> local_coords_;
  static const std::array<std::array<int, 3>, 4> faces_;

 public:
  int CountNodes() const override {
    return kNodes;
  }
  void SortNodesOnFace(const size_t *cell_nodes, size_t *face_nodes,
      int face_n_node) const final {
    if (face_n_node == 3) {
      int cnt[5] = { 0, 0, 0, 0, 1 };
      for (int f = 0; f < 3; ++f) {
        for (int c = 0; c < 4; ++c) {
          if (cell_nodes[c] == face_nodes[f]) {
            cnt[c] = 1;
            break;
          }
        }
      }
      assert(3 == std::accumulate(cnt, cnt + 5, 0));
      int i_face;
      if (cnt[0]) {
        if (cnt[1]) {
          i_face = 0;
        } else {
          assert(cnt[3]);
          i_face = 3;
        }
      } else {
        assert(cnt[2]);
        if (cnt[1]) {
          i_face = 1;
        } else {
          assert(cnt[3]);
          i_face = 2;
        }
      }
      for (int i = 0; i < 3; ++i) {
        face_nodes[i] = cell_nodes[faces_[i_face][i]];
      }
    } else {
      // the face is the bottom
      face_nodes[0] = cell_nodes[0];
      face_nodes[1] = cell_nodes[3];
      face_nodes[2] = cell_nodes[2];
      face_nodes[3] = cell_nodes[1];
    }
  }

 public:
  std::vector<Scalar> LocalToShapeFunctions(
      Scalar x_local, Scalar y_local, Scalar z_local) const override {
    auto shapes = std::vector<Scalar>(kNodes);
    for (int i = 0; i < 4; ++i) {
      auto &local_i = GetLocalCoord(i);
      auto val = (1 + local_i[X] * x_local);
          val *= (1 + local_i[Y] * y_local);
          val *= (1 + local_i[Z] * z_local);
      shapes[i] = val / 8;
    }
    shapes[4] = (1 + z_local) / 2;
    return shapes;
  }
  std::vector<Local> LocalToShapeGradients(
      Scalar x_local, Scalar y_local, Scalar z_local) const override {
    auto shapes = std::vector<Local>(kNodes);
    std::array<Scalar, kNodes> factor_xy, factor_yz, factor_zx;
    for (int i = 0; i < 4; ++i) {
      auto &local_i = GetLocalCoord(i);
      auto factor_x = (1 + local_i[X] * x_local);
      auto factor_y = (1 + local_i[Y] * y_local);
      auto factor_z = (1 + local_i[Z] * z_local);
      factor_xy[i] = factor_x * factor_y / 8;
      factor_yz[i] = factor_y * factor_z / 8;
      factor_zx[i] = factor_z * factor_x / 8;
    }
    for (int i = 0; i < 4; ++i) {
      auto &local_i = GetLocalCoord(i);
      auto &shape_i = shapes[i];
      shape_i[X] = local_i[X] * factor_yz[i];
      shape_i[Y] = local_i[Y] * factor_zx[i];
      shape_i[Z] = local_i[Z] * factor_xy[i];
    }
    shapes[4][X] = shapes[4][Y] = 0;
    shapes[4][Z] = 0.5;
    return shapes;
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
  Pyramid5(
      Global const &p0, Global const &p1, Global const &p2, Global const &p3,
      Global const &p4) {
    global_coords_[0] = p0; global_coords_[1] = p1;
    global_coords_[2] = p2; global_coords_[3] = p3;
    global_coords_[4] = p4;
    this->BuildCenter();
  }
  Pyramid5(std::initializer_list<Global> il) {
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
const std::array<typename Pyramid5<Scalar>::Local, 5>
Pyramid5<Scalar>::local_coords_{
  Pyramid5::Local(-1, -1, -1), Pyramid5::Local(+1, -1, -1),
  Pyramid5::Local(+1, +1, -1), Pyramid5::Local(-1, +1, -1),
  Pyramid5::Local(0, 0, 1)
};

template <std::floating_point Scalar>
const std::array<std::array<int, 3>, 4>
Pyramid5<Scalar>::faces_{
  /* See http://cgns.github.io/CGNS_docs_current/sids/conv.figs/pyra_5.png for node numbering.
   */
  0, 1, 4,
  1, 2, 4,
  2, 3, 4,
  3, 0, 4
};

}  // namespace lagrange
}  // namespace mini

#endif  // MINI_LAGRANGE_PYRAMID_HPP_
