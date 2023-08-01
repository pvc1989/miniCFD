//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_LAGRANGE_TETRAHEDRON_HPP_
#define MINI_LAGRANGE_TETRAHEDRON_HPP_

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
 * @brief Abstract coordinate map on tetrahedral elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar>
class Tetrahedron : public Cell<Scalar> {
  using Base = Cell<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  int CountCorners() const final {
    return 4;
  }
  const Global &center() const final {
    return center_;
  }

 protected:
  Global center_;
  void BuildCenter() final {
    Scalar a = 1.0 / 4;
    center_ = this->LocalToGlobal(a, a, a);
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
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  static constexpr int kNodes = 4;
  static constexpr int kFaces = 4;

 private:
  std::array<Global, kNodes> global_coords_;
  static const std::array<Local, kNodes> local_coords_;
  static const std::array<std::array<int, 3>, kFaces> faces_;

 public:
  int CountNodes() const final {
    return kNodes;
  }
  void SortNodesOnFace(const size_t *cell_nodes, size_t *face_nodes,
      int face_n_node) const final {
    assert(3 == face_n_node);
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
      Scalar x_local, Scalar y_local, Scalar z_local) const final {
    return {
      x_local, y_local, z_local, 1.0 - x_local - y_local - z_local
    };
  }
  std::vector<Local> LocalToShapeGradients(
      Scalar x_local, Scalar y_local, Scalar z_local) const final {
    return {
      Local(1, 0, 0), Local(0, 1, 0), Local(0, 0, 1),
      Local(-1, -1, -1)
    };
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
  Tetrahedron4(
      Global const &p0, Global const &p1,
      Global const &p2, Global const &p3) {
    global_coords_[0] = p0; global_coords_[1] = p1;
    global_coords_[2] = p2; global_coords_[3] = p3;
    this->BuildCenter();
  }

  friend void lagrange::Build(Tetrahedron4 *,
      std::initializer_list<Global>);
  Tetrahedron4(std::initializer_list<Global> il) {
    lagrange::Build(this, il);
  }
};
// initialization of static const members:
template <std::floating_point Scalar>
const std::array<typename Tetrahedron4<Scalar>::Local, 4>
Tetrahedron4<Scalar>::local_coords_{
  Tetrahedron4::Local(1, 0, 0), Tetrahedron4::Local(0, 1, 0),
  Tetrahedron4::Local(0, 0, 1), Tetrahedron4::Local(0, 0, 0)
};

template <std::floating_point Scalar>
const std::array<std::array<int, 3>, 4>
Tetrahedron4<Scalar>::faces_{
  /* See http://cgns.github.io/CGNS_docs_current/sids/conv.figs/tetra_4.png for node numbering.
   * Faces can be distinguished by the sum of the three minimum node ids.
   */
  0, 2, 1/* 3 */, 0, 1, 3/* 4 */, 0, 3, 2/* 5 */, 1, 2, 3/* 6 */
};


}  // namespace lagrange
}  // namespace mini

#endif  // MINI_LAGRANGE_TETRAHEDRON_HPP_
