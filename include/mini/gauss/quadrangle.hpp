//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_GAUSS_QUADRANGLE_HPP_
#define MINI_GAUSS_QUADRANGLE_HPP_

#include <cmath>
#include <concepts>

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
 * @tparam kDimensions  Dimension of the physical space.
 * @tparam Qx  Number of qudrature points in the \f$\xi\f$ direction.
 * @tparam Qy  Number of qudrature points in the \f$\eta\f$ direction.
 */
template <std::floating_point Scalar, int kDimensions, int Qx = 4, int Qy = 4>
class Quadrangle : public Face<Scalar, kDimensions> {
  static constexpr int D = kDimensions;

  using GaussX = GaussLegendre<Scalar, Qx>;
  using GaussY = GaussLegendre<Scalar, Qy>;

 public:
  using Lagrange = lagrange::Quadrangle<Scalar, kDimensions>;
  using Real = typename Lagrange::Real;
  using LocalCoord = typename Lagrange::LocalCoord;
  using GlobalCoord = typename Lagrange::GlobalCoord;
  using Jacobian = typename Lagrange::Jacobian;
  using Frame = typename Lagrange::Frame;

 private:
  static const std::array<LocalCoord, Qx * Qy> local_coords_;
  static const std::array<Scalar, Qx * Qy> local_weights_;
  std::array<GlobalCoord, Qx * Qy> global_coords_;
  std::array<Scalar, Qx * Qy> global_weights_;
  std::array<Frame, Qx * Qy> normal_frames_;
  Lagrange const *lagrange_;
  Scalar area_;

 public:
  int CountQuadraturePoints() const override {
    return Qx * Qy;
  }

 private:
  void BuildQuadraturePoints() {
    int n = CountQuadraturePoints();
    area_ = 0.0;
    for (int i = 0; i < n; ++i) {
      auto mat_j = lagrange().LocalToJacobian(GetLocalCoord(i));
      auto det_j = this->CellDim() < this->PhysDim()
          ? std::sqrt((mat_j.transpose() * mat_j).determinant())
          : mat_j.determinant();
      global_weights_[i] = local_weights_[i] * det_j;
      area_ += global_weights_[i];
      global_coords_[i] = lagrange().LocalToGlobal(GetLocalCoord(i));
    }
  }
  static constexpr auto BuildLocalCoords() {
    std::array<LocalCoord, Qx * Qy> points;
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
  GlobalCoord const &GetGlobalCoord(int i) const override {
    return global_coords_[i];
  }
  Scalar const &GetGlobalWeight(int i) const override {
    return global_weights_[i];
  }
  LocalCoord const &GetLocalCoord(int i) const override {
    return local_coords_[i];
  }
  Scalar const &GetLocalWeight(int i) const override {
    return local_weights_[i];
  }
  const Frame &GetNormalFrame(int i) const override {
    return normal_frames_[i];
  }
  Frame &GetNormalFrame(int i) override {
    return normal_frames_[i];
  }

 public:
  explicit Quadrangle(Lagrange const &lagrange)
      : lagrange_(&lagrange) {
    BuildQuadraturePoints();
    NormalFrameBuilder<Scalar, kDimensions>::Build(this);
  }

  const Lagrange &lagrange() const override {
    return *lagrange_;
  }

  Scalar area() const override {
    return area_;
  }
};

template <std::floating_point Scalar, int D, int Qx, int Qy>
std::array<typename Quadrangle<Scalar, D, Qx, Qy>::LocalCoord, Qx * Qy> const
Quadrangle<Scalar, D, Qx, Qy>::local_coords_
    = Quadrangle<Scalar, D, Qx, Qy>::BuildLocalCoords();

template <std::floating_point Scalar, int D, int Qx, int Qy>
std::array<Scalar, Qx * Qy> const
Quadrangle<Scalar, D, Qx, Qy>::local_weights_
    = Quadrangle<Scalar, D, Qx, Qy>::BuildLocalWeights();

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_QUADRANGLE_HPP_
