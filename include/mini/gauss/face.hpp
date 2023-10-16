//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_GAUSS_FACE_HPP_
#define MINI_GAUSS_FACE_HPP_

#include <concepts>

#include "mini/lagrange/face.hpp"
#include "mini/gauss/element.hpp"
#include "mini/gauss/function.hpp"

namespace mini {
namespace gauss {

template <std::floating_point Scalar, int kPhysDim>
class Face;

template <std::floating_point Scalar, int kPhysDim>
struct _NormalFrameBuilder {
  static void Build(Face<Scalar, kPhysDim> *face) {
  }
};

/**
 * @brief Abstract numerical integrators on surface elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam kPhysDim  Dimension of the physical space.
 */
template <std::floating_point Scalar, int kPhysDim>
class Face : public Element<Scalar, kPhysDim, 2> {
  static constexpr int D = kPhysDim;
  static_assert(D == 2 || D == 3);

 public:
  using Lagrange = lagrange::Face<Scalar, kPhysDim>;
  using Real = typename Lagrange::Real;
  using Local = typename Lagrange::Local;
  using Global = typename Lagrange::Global;
  using Jacobian = typename Lagrange::Jacobian;
  using Frame = typename Lagrange::Frame;

  virtual ~Face() noexcept = default;
  virtual const Frame &GetNormalFrame(int i) const = 0;
  virtual Frame &GetNormalFrame(int i) = 0;
  virtual Scalar area() const = 0;

  /**
   * @brief Get a reference to the lagrange::Face object it uses for coordinate mapping.
   * 
   * @return const Lagrange &  Reference to the lagrange::Face object it uses for coordinate mapping.
   */
  virtual const Lagrange &lagrange() const = 0;
};

template <std::floating_point Scalar>
struct _NormalFrameBuilder<Scalar, 3> {
  static void Build(Face<Scalar, 3> *face) {
    int n = face->CountPoints();
    for (int i = 0; i < n; ++i) {
      auto &local_i = face->GetLocalCoord(i);
      face->GetNormalFrame(i) = face->lagrange().LocalToNormalFrame(local_i);
    }
  }
};

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_FACE_HPP_
