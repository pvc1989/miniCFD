//  Copyright 2023 PEI Weicheng
#ifndef MINI_LAGRANGE_TRIANGLE_HPP_
#define MINI_LAGRANGE_TRIANGLE_HPP_

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstring>
#include <type_traits>
#include <utility>

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
  using typename Base::LocalCoord;
  using typename Base::GlobalCoord;
  using typename Base::Jacobian;

  int CountCorners() const override final {
    return 3;
  }
  const GlobalCoord &center() const override final {
    return center_;
  }

 protected:
  GlobalCoord center_;
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
  using typename Base::LocalCoord;
  using typename Base::GlobalCoord;
  using typename Base::Jacobian;

  static constexpr int kNodes = 3;

 private:
  std::array<GlobalCoord, kNodes> global_coords_;
  static const std::array<LocalCoord, kNodes> local_coords_;

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
  std::vector<LocalCoord> LocalToShapeGradients(Scalar x_local, Scalar y_local)
      const override {
    return {
      LocalCoord(1, 0), LocalCoord(0, 1), LocalCoord(-1, -1)
    };
  }

 public:
  GlobalCoord const &GetGlobalCoord(int q) const override {
    return global_coords_[q];
  }
  LocalCoord const &GetLocalCoord(int q) const override {
    return local_coords_[q];
  }

 public:
  Triangle3(
      GlobalCoord const &p0, GlobalCoord const &p1,
      GlobalCoord const &p2) {
    global_coords_[0] = p0; global_coords_[1] = p1;
    global_coords_[2] = p2;
    this->BuildCenter();
  }
  Triangle3(std::initializer_list<GlobalCoord> il) {
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
const std::array<typename Triangle3<Scalar, kPhysDim>::LocalCoord, 3>
Triangle3<Scalar, kPhysDim>::local_coords_{
  Triangle3::LocalCoord(1, 0), Triangle3::LocalCoord(0, 1),
  Triangle3::LocalCoord(0, 0)
};

}  // namespace lagrange
}  // namespace mini

#endif  // MINI_LAGRANGE_TRIANGLE_HPP_
