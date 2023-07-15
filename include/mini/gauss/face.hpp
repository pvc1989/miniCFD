//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_GAUSS_FACE_HPP_
#define MINI_GAUSS_FACE_HPP_

#include <concepts>

#include "mini/lagrange/face.hpp"
#include "mini/gauss/element.hpp"
#include "mini/gauss/function.hpp"

namespace mini {
namespace gauss {

template <std::floating_point Scalar, int kDimensions>
class Face;

template <std::floating_point Scalar, int kDimensions>
struct NormalFrameBuilder {
  static void Build(Face<Scalar, kDimensions> *face) {
  }
};

/**
 * @brief Abstract numerical integrators on surface elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam kDimensions  Dimension of the physical space.
 */
template <std::floating_point Scalar, int kDimensions>
class Face {
  static constexpr int D = kDimensions;
  static_assert(D == 2 || D == 3);

 public:
  using Lagrange = lagrange::Face<Scalar, kDimensions>;
  using Real = typename Lagrange::Real;
  using LocalCoord = typename Lagrange::LocalCoord;
  using GlobalCoord = typename Lagrange::GlobalCoord;
  using Jacobian = typename Lagrange::Jacobian;
  using Frame = typename Lagrange::Frame;

  virtual ~Face() noexcept = default;
  virtual int CountQuadraturePoints() const = 0;
  virtual const LocalCoord &GetLocalCoord(int i) const = 0;
  virtual const GlobalCoord &GetGlobalCoord(int i) const = 0;
  virtual Real const &GetLocalWeight(int i) const = 0;
  virtual Real const &GetGlobalWeight(int i) const = 0;
  virtual const Frame &GetNormalFrame(int i) const = 0;
  virtual Frame &GetNormalFrame(int i) = 0;
  virtual Scalar area() const = 0;

  virtual const Lagrange &lagrange() const = 0;

  int CountCorners() const {
    return lagrange().CountCorners();
  }
  int CountNodes() const {
    return lagrange().CountNodes();
  }

  const GlobalCoord &center() const {
    return lagrange().center();
  }

  GlobalCoord LocalToGlobal(Scalar x_local, Scalar y_local)
      const {
    return lagrange().LocalToGlobal(x_local, y_local);
  }
  GlobalCoord LocalToGlobal(const LocalCoord &xy) const {
    return LocalToGlobal(xy[X], xy[Y]);
  }

  Jacobian LocalToJacobian(Scalar x_local, Scalar y_local)
      const {
    return lagrange().LocalToJacobian(x_local, y_local);
  }
  Jacobian LocalToJacobian(const LocalCoord &xy) const {
    return LocalToJacobian(xy[X], xy[Y]);
  }

  static constexpr int CellDim() {
    return 2;
  }
  static constexpr int PhysDim() {
    return D;
  }
};

template <std::floating_point Scalar>
struct NormalFrameBuilder<Scalar, 3> {
  static void Build(Face<Scalar, 3> *face) {
    int n = face->CountQuadraturePoints();
    for (int i = 0; i < n; ++i) {
      auto &local_i = face->GetLocalCoord(i);
      face->GetNormalFrame(i) = face->lagrange().LocalToNormalFrame(local_i);
    }
  }
};

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_FACE_HPP_
