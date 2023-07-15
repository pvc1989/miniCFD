//  Copyright 2023 PEI Weicheng
#ifndef MINI_LAGRANGE_QUADRANGLE_HPP_
#define MINI_LAGRANGE_QUADRANGLE_HPP_

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
 * @brief Abstract coordinate map on quadrilateral elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar, int kDimensions>
class Quadrangle : public Face<Scalar, kDimensions> {
  using Base = Face<Scalar, kDimensions>;

 public:
  using typename Base::Real;
  using typename Base::LocalCoord;
  using typename Base::GlobalCoord;
  using typename Base::Jacobian;

  int CountCorners() const override final {
    return 4;
  }
  const GlobalCoord &center() const override final {
    return center_;
  }

 protected:
  GlobalCoord center_;
  void BuildCenter() {
    Scalar a = 0;
    center_ = this->LocalToGlobal(a, a);
  }
};

/**
 * @brief Coordinate map on 4-node quadrilateral elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 */
template <std::floating_point Scalar, int kDimensions>
class Quadrangle4 : public Quadrangle<Scalar, kDimensions> {
  using Base = Quadrangle<Scalar, kDimensions>;

 public:
  using typename Base::Real;
  using typename Base::LocalCoord;
  using typename Base::GlobalCoord;
  using typename Base::Jacobian;

  static constexpr int kNodes = 4;

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
    auto shapes = std::vector<Scalar>(kNodes);
    for (int i = 0; i < kNodes; ++i) {
      auto &local_i = GetLocalCoord(i);
      auto val = (1 + local_i[X] * x_local) * (1 + local_i[Y] * y_local);
      shapes[i] = val / 4;
    }
    return shapes;
  }
  std::vector<LocalCoord> LocalToShapeGradients(Scalar x_local, Scalar y_local)
      const override {
    auto shapes = std::vector<LocalCoord>(kNodes);
    std::array<Scalar, kNodes> factor_x, factor_y;
    for (int i = 0; i < kNodes; ++i) {
      auto &local_i = GetLocalCoord(i);
      factor_x[i] = (1 + local_i[X] * x_local) / 4;
      factor_y[i] = (1 + local_i[Y] * y_local) / 4;
    }
    for (int i = 0; i < kNodes; ++i) {
      auto &local_i = GetLocalCoord(i);
      auto &shape_i = shapes[i];
      shape_i[X] = local_i[X] * factor_y[i];
      shape_i[Y] = local_i[Y] * factor_x[i];
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
  Quadrangle4(
      GlobalCoord const &p0, GlobalCoord const &p1,
      GlobalCoord const &p2, GlobalCoord const &p3) {
    global_coords_[0] = p0; global_coords_[1] = p1;
    global_coords_[2] = p2; global_coords_[3] = p3;
    this->BuildCenter();
  }
  Quadrangle4(std::initializer_list<GlobalCoord> il) {
    assert(il.size() == kNodes);
    auto p = il.begin();
    for (int i = 0; i < kNodes; ++i) {
      global_coords_[i] = p[i];
    }
    this->BuildCenter();
  }
};
// initialization of static const members:
template <std::floating_point Scalar, int kDimensions>
const std::array<typename Quadrangle4<Scalar, kDimensions>::LocalCoord, 4>
Quadrangle4<Scalar, kDimensions>::local_coords_{
  Quadrangle4::LocalCoord(-1, -1), Quadrangle4::LocalCoord(+1, -1),
  Quadrangle4::LocalCoord(+1, +1), Quadrangle4::LocalCoord(-1, +1)
};

}  // namespace lagrange
}  // namespace mini

#endif  // MINI_LAGRANGE_QUADRANGLE_HPP_
