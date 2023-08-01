//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_GAUSS_QUADRANGLE_HPP_
#define MINI_GAUSS_QUADRANGLE_HPP_

#include <concepts>

#include <cmath>

#include "mini/algebra/eigen.hpp"

#include "mini/gauss/gauss.hpp"
#include "mini/gauss/face.hpp"
#include "mini/lagrange/quadrangle.hpp"

namespace mini {
namespace gauss {

/**
 * @brief Numerical integrators on quadrilateral elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam kPhysDim  Dimension of the physical space.
 * @tparam Qx  Number of qudrature points in the \f$\xi\f$ direction.
 * @tparam Qy  Number of qudrature points in the \f$\eta\f$ direction.
 */
template <std::floating_point Scalar, int kPhysDim, int Qx = 4, int Qy = 4>
class Quadrangle : public Face<Scalar, kPhysDim> {
  static constexpr int D = kPhysDim;

  using GaussX = GaussLegendre<Scalar, Qx>;
  using GaussY = GaussLegendre<Scalar, Qy>;

 public:
  using Lagrange = lagrange::Quadrangle<Scalar, kPhysDim>;
  using Real = typename Lagrange::Real;
  using Local = typename Lagrange::Local;
  using Global = typename Lagrange::Global;
  using Jacobian = typename Lagrange::Jacobian;
  using Frame = typename Lagrange::Frame;

 private:
  static const std::array<Local, Qx * Qy> local_coords_;
  static const std::array<Scalar, Qx * Qy> local_weights_;
  std::array<Global, Qx * Qy> global_coords_;
  std::array<Scalar, Qx * Qy> global_weights_;
  std::array<Frame, Qx * Qy> normal_frames_;
  Lagrange const *lagrange_;
  Scalar area_;

 public:
  int CountPoints() const final {
    return Qx * Qy;
  }

 private:
  static constexpr auto BuildLocalCoords() {
    std::array<Local, Qx * Qy> points;
    int k = 0;
    for (int i = 0; i < Qx; ++i) {
      for (int j = 0; j < Qy; ++j) {
        points[k][0] = GaussX::points[i];
        points[k][1] = GaussY::points[j];
        k++;
      }
    }
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    std::array<Scalar, Qx * Qy> weights;
    int k = 0;
    for (int i = 0; i < Qx; ++i) {
      for (int j = 0; j < Qy; ++j) {
        weights[k++] = GaussX::weights[i] * GaussY::weights[j];
      }
    }
    return weights;
  }

 public:
  const Global &GetGlobalCoord(int i) const final {
    assert(0 <= i && i < CountPoints());
    return global_coords_[i];
  }
  const Scalar &GetGlobalWeight(int i) const final {
    assert(0 <= i && i < CountPoints());
    return global_weights_[i];
  }
  const Local &GetLocalCoord(int i) const final {
    assert(0 <= i && i < CountPoints());
    return local_coords_[i];
  }
  const Scalar &GetLocalWeight(int i) const final {
    assert(0 <= i && i < CountPoints());
    return local_weights_[i];
  }

 protected:
  Global &GetGlobalCoord(int i) final {
    assert(0 <= i && i < CountPoints());
    return global_coords_[i];
  }
  Scalar &GetGlobalWeight(int i) final {
    assert(0 <= i && i < CountPoints());
    return global_weights_[i];
  }

 public:
  const Frame &GetNormalFrame(int i) const final {
    assert(0 <= i && i < CountPoints());
    return normal_frames_[i];
  }
  Frame &GetNormalFrame(int i) final {
    assert(0 <= i && i < CountPoints());
    return normal_frames_[i];
  }

 public:
  explicit Quadrangle(Lagrange const &lagrange)
      : lagrange_(&lagrange) {
    area_ = this->BuildQuadraturePoints();
    NormalFrameBuilder<Scalar, kPhysDim>::Build(this);
  }

  const Lagrange &lagrange() const final {
    return *lagrange_;
  }

  Scalar area() const final {
    return area_;
  }
};

template <std::floating_point Scalar, int D, int Qx, int Qy>
std::array<typename Quadrangle<Scalar, D, Qx, Qy>::Local, Qx * Qy> const
Quadrangle<Scalar, D, Qx, Qy>::local_coords_
    = Quadrangle<Scalar, D, Qx, Qy>::BuildLocalCoords();

template <std::floating_point Scalar, int D, int Qx, int Qy>
std::array<Scalar, Qx * Qy> const
Quadrangle<Scalar, D, Qx, Qy>::local_weights_
    = Quadrangle<Scalar, D, Qx, Qy>::BuildLocalWeights();

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_QUADRANGLE_HPP_
