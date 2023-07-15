//  Copyright 2023 PEI Weicheng
#ifndef MINI_LAGRANGE_FACE_HPP_
#define MINI_LAGRANGE_FACE_HPP_

#include <algorithm>
#include <array>
#include <concepts>
#include <type_traits>

#include "mini/algebra/eigen.hpp"
#include "mini/lagrange/element.hpp"

namespace mini {
namespace lagrange {

template <std::floating_point Scalar, int kDimensions>
class Face;

template <std::floating_point Scalar, int kDimensions>
struct NormalFrameBuilder {
  using Frame = typename Face<Scalar, kDimensions>::Frame;
  static Frame Build(const Face<Scalar, kDimensions> &face,
      Scalar x_local, Scalar y_local) {
    return Frame();
  }
};

/**
 * @brief Abstract coordinate map on surface elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam kDimensions  Dimension of the physical space.
 */
template <std::floating_point Scalar, int kDimensions>
class Face {
  static constexpr int D = kDimensions;
  static_assert(D == 2 || D == 3);

 public:
  using Real = Scalar;
  using LocalCoord = algebra::Matrix<Scalar, 2, 1>;
  using GlobalCoord = algebra::Matrix<Scalar, D, 1>;
  using Jacobian = algebra::Matrix<Scalar, D, 2>;
  using Frame = std::conditional_t<D == 3, std::array<GlobalCoord, 3>, int>;

  virtual ~Face() noexcept = default;
  virtual std::vector<Scalar> LocalToShapeFunctions(Scalar, Scalar) const = 0;
  virtual std::vector<LocalCoord> LocalToShapeGradients(Scalar, Scalar) const = 0;
  virtual int CountCorners() const = 0;
  virtual int CountNodes() const = 0;
  virtual const LocalCoord &GetLocalCoord(int i) const = 0;
  virtual const GlobalCoord &GetGlobalCoord(int i) const = 0;
  virtual const GlobalCoord &center() const = 0;

  std::vector<Scalar> LocalToShapeFunctions(const LocalCoord &xy)
      const {
    return LocalToShapeFunctions(xy[X], xy[Y]);
  }
  std::vector<LocalCoord> LocalToShapeGradients(const LocalCoord &xy)
      const {
    return LocalToShapeGradients(xy[X], xy[Y]);
  }
  Frame LocalToNormalFrame(Scalar x_local, Scalar y_local) const {
    return NormalFrameBuilder<Scalar, kDimensions>
        ::Build(*this, x_local, y_local);
  }
  Frame LocalToNormalFrame(const LocalCoord &xy) const {
    return LocalToNormalFrame(xy[X], xy[Y]);
  }

  static constexpr int CellDim() {
    return 2;
  }
  static constexpr int PhysDim() {
    return D;
  }

  GlobalCoord LocalToGlobal(Scalar x_local, Scalar y_local) const {
    auto shapes = LocalToShapeFunctions(x_local, y_local);
    GlobalCoord sum = GetGlobalCoord(0) * shapes[0];
    for (int i = 1; i < CountNodes(); ++i) {
      sum += GetGlobalCoord(i) * shapes[i];
    }
    return sum;
  }
  GlobalCoord LocalToGlobal(const LocalCoord &xy) const {
    return LocalToGlobal(xy[X], xy[Y]);
  }

  Jacobian LocalToJacobian(Scalar x_local, Scalar y_local) const {
    auto shapes = LocalToShapeGradients(x_local, y_local);
    Jacobian sum = GetGlobalCoord(0) * shapes[0].transpose();
    for (int i = 1; i < CountNodes(); ++i) {
      sum += GetGlobalCoord(i) * shapes[i].transpose();
    }
    return sum;
  }
  Jacobian LocalToJacobian(const LocalCoord &xy) const {
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
