//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_LAGRANGE_TETRAHEDRON_HPP_
#define MINI_LAGRANGE_TETRAHEDRON_HPP_

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
 * @brief Abstract coordinate map on tetrahedral elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Tetrahedron : public virtual Cell<Scalar> {
  using Base = Cell<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::LocalCoord;
  using typename Base::GlobalCoord;
  using typename Base::Jacobian;

  int CountVertices() const override final {
    return 4;
  }
  GlobalCoord center() const override final {
    Scalar a = 1.0 / 4;
    return this->LocalToGlobal(a, a, a);
  }

};

/**
 * @brief Coordinate map on 4-node tetrahedral elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Tetrahedron4 : public Tetrahedron<Scalar> {
  using Base = Tetrahedron<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::LocalCoord;
  using typename Base::GlobalCoord;
  using typename Base::Jacobian;

  static constexpr int kNodes = 4;
  static constexpr int kFaces = 4;

 private:
  std::array<GlobalCoord, kNodes> global_coords_;
  static const std::array<LocalCoord, kNodes> local_coords_;
  static const std::array<std::array<int, 3>, kFaces> faces_;

 public:
  int CountNodes() const override {
    return kNodes;
  }
  template <typename T, typename U>
  static void SortNodesOnFace(const T *cell_nodes, U *face_nodes) {
    int cnt = 0, nid = 0, sum = 0;
    while (cnt < 3) {
      auto curr_node = cell_nodes[nid];
      for (int i = 0; i < 3; ++i) {
        if (face_nodes[i] == curr_node) {
          sum += nid;
          ++cnt;
          break;
        }
      }
      ++nid;
    }
    int i_face = sum - 3;
    for (int i = 0; i < 3; ++i) {
      face_nodes[i] = cell_nodes[faces_[i_face][i]];
    }
  }

 public:
  std::vector<Scalar> LocalToShapeFunctions(
      Scalar x_local, Scalar y_local, Scalar z_local) const override {
    return {
      x_local, y_local, z_local, 1.0 - x_local - y_local - z_local
    };
  }
  std::vector<LocalCoord> LocalToShapeGradients(
      Scalar x_local, Scalar y_local, Scalar z_local) const override {
    return {
      LocalCoord(1, 0, 0), LocalCoord(0, 1, 0), LocalCoord(0, 0, 1),
      LocalCoord(-1, -1, -1)
    };
  }

 public:
  GlobalCoord const &GetGlobalCoordL(int q) const override {
    return global_coords_[q];
  }
  LocalCoord const &GetLocalCoordL(int q) const override {
    return local_coords_[q];
  }

 public:
  Tetrahedron4(
      GlobalCoord const &p0, GlobalCoord const &p1,
      GlobalCoord const &p2, GlobalCoord const &p3) {
    global_coords_[0] = p0; global_coords_[1] = p1;
    global_coords_[2] = p2; global_coords_[3] = p3;
  }
  Tetrahedron4(std::initializer_list<GlobalCoord> il) {
    assert(il.size() == kNodes);
    auto p = il.begin();
    for (int i = 0; i < kNodes; ++i) {
      global_coords_[i] = p[i];
    }
  }
};
// initialization of static const members:
template <std::floating_point Scalar>
const std::array<typename Tetrahedron4<Scalar>::LocalCoord, 4>
Tetrahedron4<Scalar>::local_coords_{
  Tetrahedron4::LocalCoord(1, 0, 0), Tetrahedron4::LocalCoord(0, 1, 0),
  Tetrahedron4::LocalCoord(0, 0, 1), Tetrahedron4::LocalCoord(0, 0, 0)
};

template <std::floating_point Scalar>
const std::array<std::array<int, 3>, 4>
Tetrahedron4<Scalar>::faces_{
  // Faces can be distinguished by the sum of the three minimum node ids.
  0, 2, 1/* 3 */, 0, 1, 3/* 4 */, 2, 0, 3/* 5 */, 1, 2, 3/* 6 */
};


}  // namespace lagrange
}  // namespace mini

#endif  // MINI_LAGRANGE_TETRAHEDRON_HPP_
