//  Copyright 2023 PEI Weicheng
#ifndef MINI_LAGRANGE_FACE_HPP_
#define MINI_LAGRANGE_FACE_HPP_

#include <algorithm>
#include <array>
#include <concepts>

#include "mini/algebra/eigen.hpp"

namespace mini {
namespace lagrange {

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

  static constexpr int X{0}, Y{1}, Z{2};

 public:
  using Real = Scalar;
  using LocalCoord = algebra::Matrix<Scalar, 2, 1>;
  using GlobalCoord = algebra::Matrix<Scalar, D, 1>;
  using Jacobian = algebra::Matrix<Scalar, D, 2>;

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
  std::array<GlobalCoord, D> LocalToNormalFrame(Scalar x_local, Scalar y_local)
      const {
    std::array<GlobalCoord, D> frame;
    GlobalCoord &normal = frame[X], &tangent = frame[Y];
    GlobalCoord &bitangent = frame.back();
    auto jacobian = LocalToJacobian(x_local, y_local);
    normal = jacobian.col(X).cross(jacobian.col(Y)).normalized();
    tangent = jacobian.col(Y).normalized();
    bitangent = (D == 3 ? normal.cross(tangent) : bitangent);
    return frame;
  }
  std::array<GlobalCoord, D> LocalToNormalFrame(const LocalCoord &xy) const {
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

}  // namespace lagrange
}  // namespace mini

#endif  // MINI_LAGRANGE_FACE_HPP_
