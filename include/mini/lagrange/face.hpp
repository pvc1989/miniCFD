//  Copyright 2023 PEI Weicheng
#ifndef MINI_LAGRANGE_FACE_HPP_
#define MINI_LAGRANGE_FACE_HPP_

#include <algorithm>
#include <array>
#include <concepts>
#include <type_traits>

#include "mini/lagrange/element.hpp"

namespace mini {
namespace lagrange {

template <std::floating_point Scalar, int kPhysDim>
class Face;

template <std::floating_point Scalar, int kPhysDim>
struct NormalFrameBuilder {
  using Frame = typename Face<Scalar, kPhysDim>::Frame;
  static Frame Build(const Face<Scalar, kPhysDim> &face,
      Scalar x_local, Scalar y_local) {
    return Frame();
  }
};

/**
 * @brief Abstract coordinate map on surface elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam kPhysDim  Dimension of the physical space.
 */
template <std::floating_point Scalar, int kPhysDim>
class Face : public Element<Scalar, kPhysDim, 2> {
  static constexpr int D = kPhysDim;
  static_assert(D == 2 || D == 3);

  using Base = Element<Scalar, kPhysDim, 2>;

 public:
  using Real = typename Base::Real;
  using Local = typename Base::Local;
  using Global = typename Base::Global;
  using Jacobian = typename Base::Jacobian;
  using Frame = std::conditional_t<D == 3, std::array<Global, 3>, int>;

  virtual std::vector<Scalar> LocalToShapeFunctions(Scalar, Scalar) const = 0;
  virtual std::vector<Local> LocalToShapeGradients(Scalar, Scalar) const = 0;

  std::vector<Scalar> LocalToShapeFunctions(const Local &xy)
      const override {
    return LocalToShapeFunctions(xy[X], xy[Y]);
  }
  std::vector<Local> LocalToShapeGradients(const Local &xy)
      const override {
    return LocalToShapeGradients(xy[X], xy[Y]);
  }
  Frame LocalToNormalFrame(Scalar x_local, Scalar y_local) const {
    return NormalFrameBuilder<Scalar, kPhysDim>
        ::Build(*this, x_local, y_local);
  }
  Frame LocalToNormalFrame(const Local &xy) const {
    return LocalToNormalFrame(xy[X], xy[Y]);
  }

  Global LocalToGlobal(Scalar x_local, Scalar y_local) const {
    auto shapes = LocalToShapeFunctions(x_local, y_local);
    Global sum = this->GetGlobalCoord(0) * shapes[0];
    for (int i = 1, n = this->CountNodes(); i < n; ++i) {
      sum += this->GetGlobalCoord(i) * shapes[i];
    }
    return sum;
  }
  Global LocalToGlobal(const Local &xy) const override {
    return LocalToGlobal(xy[X], xy[Y]);
  }

  Jacobian LocalToJacobian(Scalar x_local, Scalar y_local) const {
    auto shapes = LocalToShapeGradients(x_local, y_local);
    Jacobian sum = this->GetGlobalCoord(0) * shapes[0].transpose();
    for (int i = 1, n = this->CountNodes(); i < n; ++i) {
      sum += this->GetGlobalCoord(i) * shapes[i].transpose();
    }
    return sum;
  }
  Jacobian LocalToJacobian(const Local &xy) const override {
    return LocalToJacobian(xy[X], xy[Y]);
  }

};

template <std::floating_point Scalar>
struct NormalFrameBuilder<Scalar, 3> {
  using Frame = typename Face<Scalar, 3>::Frame;
  static Frame Build(const Face<Scalar, 3> &face,
      Scalar x_local, Scalar y_local) {
    Frame frame;
    auto &normal = frame[X], &tangent = frame[Y], &bitangent = frame[Z];
    auto jacobian = face.LocalToJacobian(x_local, y_local);
    normal = jacobian.col(X).cross(jacobian.col(Y)).normalized();
    tangent = jacobian.col(X).normalized();
    bitangent = normal.cross(tangent);
    return frame;
  }
};

}  // namespace lagrange
}  // namespace mini

#endif  // MINI_LAGRANGE_FACE_HPP_
