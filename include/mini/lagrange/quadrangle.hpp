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
template <std::floating_point Scalar, int kPhysDim>
class Quadrangle : public Face<Scalar, kPhysDim> {
  using Base = Face<Scalar, kPhysDim>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  int CountCorners() const override final {
    return 4;
  }
  const Global &center() const override final {
    return center_;
  }

 protected:
  Global center_;
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
template <std::floating_point Scalar, int kPhysDim>
class Quadrangle4 : public Quadrangle<Scalar, kPhysDim> {
  using Base = Quadrangle<Scalar, kPhysDim>;

 public:
  using typename Base::Real;
  using typename Base::Local;
  using typename Base::Global;
  using typename Base::Jacobian;

  static constexpr int kNodes = 4;

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
    auto shapes = std::vector<Scalar>(kNodes);
    for (int i = 0; i < kNodes; ++i) {
      auto &local_i = GetLocalCoord(i);
      auto val = (1 + local_i[X] * x_local) * (1 + local_i[Y] * y_local);
      shapes[i] = val / 4;
    }
    return shapes;
  }
  std::vector<Local> LocalToShapeGradients(Scalar x_local, Scalar y_local)
      const override {
    auto shapes = std::vector<Local>(kNodes);
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
  Global const &GetGlobalCoord(int i) const override {
    assert(0 <= i && i < CountNodes());
    return global_coords_[i];
  }
  Local const &GetLocalCoord(int i) const override {
    assert(0 <= i && i < CountNodes());
    return local_coords_[i];
  }

 public:
  Quadrangle4(
      Global const &p0, Global const &p1,
      Global const &p2, Global const &p3) {
    global_coords_[0] = p0; global_coords_[1] = p1;
    global_coords_[2] = p2; global_coords_[3] = p3;
    this->BuildCenter();
  }
  Quadrangle4(std::initializer_list<Global> il) {
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
const std::array<typename Quadrangle4<Scalar, kPhysDim>::Local, 4>
Quadrangle4<Scalar, kPhysDim>::local_coords_{
  Quadrangle4::Local(-1, -1), Quadrangle4::Local(+1, -1),
  Quadrangle4::Local(+1, +1), Quadrangle4::Local(-1, +1)
};

}  // namespace lagrange
}  // namespace mini

#endif  // MINI_LAGRANGE_QUADRANGLE_HPP_
