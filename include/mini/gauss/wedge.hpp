//  Copyright 2023 PEI Weicheng
#ifndef MINI_GAUSS_WEDGE_HPP_
#define MINI_GAUSS_WEDGE_HPP_

#include <concepts>

#include <cassert>
#include <cmath>
#include <cstring>

#include <algorithm>
#include <type_traits>

#include "mini/gauss/gauss.hpp"
#include "mini/gauss/cell.hpp"
#include "mini/gauss/triangle.hpp"
#include "mini/lagrange/wedge.hpp"

namespace mini {
namespace gauss {
/**
 * @brief Numerical integrators on hexahedral elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam Qt  Number of qudrature points in each layer of triangle.
 * @tparam Qz  Number of qudrature points in the \f$\zeta\f$ direction.
 */
template <std::floating_point Scalar, int Qt, int Qz>
class Wedge : public Cell<Scalar> {
  static constexpr int kPoints = Qt * Qz;
  using GaussT = TriangleBuilder<Scalar, 2, Qt>;
  using GaussZ = GaussLegendre<Scalar, Qz>;

 public:
  using Lagrange = lagrange::Wedge<Scalar>;
  using Real = typename Lagrange::Real;
  using Local = typename Lagrange::Local;
  using Global = typename Lagrange::Global;
  using Jacobian = typename Lagrange::Jacobian;

 private:
  static const std::array<Local, Qt * Qz> local_coords_;
  static const std::array<Scalar, Qt * Qz> local_weights_;
  std::array<Global, kPoints> global_coords_;
  std::array<Scalar, kPoints> global_weights_;
  Lagrange const *lagrange_;
  Scalar volume_;

 public:
  int CountPoints() const final {
    return kPoints;
  }

 private:
  static constexpr auto BuildLocalCoords() {
    std::array<Local, kPoints> points;
    auto triangle_points = GaussT::BuildLocalCoords();
    int n = 0;
    for (int i = 0; i < Qt; ++i) {
      for (int k = 0; k < Qz; ++k) {
        points[n][X] = triangle_points[i][X];
        points[n][Y] = triangle_points[i][Y];
        points[n][Z] = GaussZ::points[k];
        n++;
      }
    }
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    std::array<Scalar, Qt * Qz> weights;
    auto triangle_weights = GaussT::BuildLocalWeights();
    int n = 0;
    for (int i = 0; i < Qt; ++i) {
      for (int k = 0; k < Qz; ++k) {
        weights[n++] = triangle_weights[i] * GaussZ::weights[k];
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
  explicit Wedge(Lagrange const &lagrange)
      : lagrange_(&lagrange) {
    volume_ = this->BuildQuadraturePoints();
  }
  Wedge(const Wedge &) = default;
  Wedge &operator=(const Wedge &) = default;
  Wedge(Wedge &&) noexcept = default;
  Wedge &operator=(Wedge &&) noexcept = default;
  virtual ~Wedge() noexcept = default;

  const Lagrange &lagrange() const final {
    return *lagrange_;
  }

  Scalar volume() const final {
    return volume_;
  }
};

template <std::floating_point Scalar, int Qt, int Qz>
std::array<typename Wedge<Scalar, Qt, Qz>::Local, Qt * Qz> const
Wedge<Scalar, Qt, Qz>::local_coords_
    = Wedge<Scalar, Qt, Qz>::BuildLocalCoords();

template <std::floating_point Scalar, int Qt, int Qz>
std::array<Scalar, Qt * Qz> const
Wedge<Scalar, Qt, Qz>::local_weights_
    = Wedge<Scalar, Qt, Qz>::BuildLocalWeights();

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_WEDGE_HPP_
