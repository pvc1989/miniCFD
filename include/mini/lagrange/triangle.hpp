//  Copyright 2023 PEI Weicheng
#ifndef MINI_LAGRANGE_TRIANGLE_HPP_
#define MINI_LAGRANGE_TRIANGLE_HPP_

#include <concepts>

#include <cassert>
#include <cmath>
#include <cstring>

#include <algorithm>
#include <array>
#include <type_traits>
#include <utility>
#include <vector>

#include "mini/lagrange/face.hpp"

namespace mini {
namespace lagrange {

/**
 * @brief Abstract coordinate map on triangular elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar, int kPhysDim>
class Triangle : public Face<Scalar, kPhysDim> {
  using Base = Face<Scalar, kPhysDim>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  int CountCorners() const final {
    return 3;
  }
  const Global &center() const final {
    return center_;
  }

 protected:
  Global center_;
  void BuildCenter() {
    Scalar a = 1.0 / 3;
    center_ = this->LocalToGlobal(a, a);
  }
};

/**
 * @brief Coordinate map on 3-node triangular elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar, int kPhysDim>
class Triangle3 : public Triangle<Scalar, kPhysDim> {
  using Base = Triangle<Scalar, kPhysDim>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  static constexpr int kNodes = 3;

 private:
  std::array<Global, kNodes> global_coords_;
  static const std::array<Local, kNodes> local_coords_;

 public:
  int CountNodes() const override {
    return kNodes;
  }

 public:
  std::vector<Scalar> LocalToShapeFunctions(Scalar x_local, Scalar y_local)
      const override {
    return {
      x_local, y_local, 1.0 - x_local - y_local
    };
  }
  std::vector<Local> LocalToShapeGradients(Scalar x_local, Scalar y_local)
      const override {
    return {
      Local(1, 0), Local(0, 1), Local(-1, -1)
    };
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
  Triangle3(
      Global const &p0, Global const &p1,
      Global const &p2) {
    global_coords_[0] = p0; global_coords_[1] = p1;
    global_coords_[2] = p2;
    this->BuildCenter();
  }
  Triangle3(std::initializer_list<Global> il) {
    assert(il.size() == kNodes);
    auto p = il.begin();
    for (int i = 0; i < kNodes; ++i) {
      global_coords_[i] = p[i];
    }
    this->BuildCenter();
  }
};
// initialization of static const members:
template <std::floating_point Scalar, int kPhysDim>
const std::array<typename Triangle3<Scalar, kPhysDim>::Local, 3>
Triangle3<Scalar, kPhysDim>::local_coords_{
  Triangle3::Local(1, 0), Triangle3::Local(0, 1),
  Triangle3::Local(0, 0)
};

}  // namespace lagrange
}  // namespace mini

#endif  // MINI_LAGRANGE_TRIANGLE_HPP_
