//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_GAUSS_HEXAHEDRON_HPP_
#define MINI_GAUSS_HEXAHEDRON_HPP_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <concepts>
#include <type_traits>

#include "mini/gauss/gauss.hpp"
#include "mini/gauss/cell.hpp"
#include "mini/lagrange/hexahedron.hpp"

namespace mini {
namespace gauss {
/**
 * @brief Numerical integrators on hexahedral elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam Qx  Number of qudrature points in the \f$\xi\f$ direction.
 * @tparam Qy  Number of qudrature points in the \f$\eta\f$ direction.
 * @tparam Qz  Number of qudrature points in the \f$\zeta\f$ direction.
 */
template <std::floating_point Scalar, int Qx = 4, int Qy = 4, int Qz = 4>
class Hexahedron : public Cell<Scalar> {
  using GaussX = GaussLegendre<Scalar, Qx>;
  using GaussY = GaussLegendre<Scalar, Qy>;
  using GaussZ = GaussLegendre<Scalar, Qz>;

 public:
  using Lagrange = lagrange::Hexahedron<Scalar>;
  using Real = typename Lagrange::Real;
  using Local = typename Lagrange::Local;
  using Global = typename Lagrange::Global;
  using Jacobian = typename Lagrange::Jacobian;

 private:
  static const std::array<Local, Qx * Qy * Qz> local_coords_;
  static const std::array<Scalar, Qx * Qy * Qz> local_weights_;
  std::array<Global, Qx * Qy * Qz> global_coords_;
  std::array<Scalar, Qx * Qy * Qz> global_weights_;
  Lagrange const *lagrange_;
  Scalar volume_;

 public:
  int CountPoints() const override {
    return Qx * Qy * Qz;
  }

 private:
  static constexpr auto BuildLocalCoords() {
    std::array<Local, Qx * Qy * Qz> points;
    int n = 0;
    for (int i = 0; i < Qx; ++i) {
      for (int j = 0; j < Qy; ++j) {
        for (int k = 0; k < Qz; ++k) {
          points[n][X] = GaussX::points[i];
          points[n][Y] = GaussY::points[j];
          points[n][Z] = GaussZ::points[k];
          n++;
        }
      }
    }
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    std::array<Scalar, Qx * Qy * Qz> weights;
    int n = 0;
    for (int i = 0; i < Qx; ++i) {
      for (int j = 0; j < Qy; ++j) {
        for (int k = 0; k < Qz; ++k) {
          weights[n++] = GaussX::weights[i] * GaussY::weights[j]
              * GaussZ::weights[k];
        }
      }
    }
    return weights;
  }

 public:
  const Global &GetGlobalCoord(int i) const override {
    assert(0 <= i && i < CountPoints());
    return global_coords_[i];
  }
  const Scalar &GetGlobalWeight(int i) const override {
    assert(0 <= i && i < CountPoints());
    return global_weights_[i];
  }
  const Local &GetLocalCoord(int i) const override {
    assert(0 <= i && i < CountPoints());
    return local_coords_[i];
  }
  const Scalar &GetLocalWeight(int i) const override {
    assert(0 <= i && i < CountPoints());
    return local_weights_[i];
  }

 protected:
  Global &GetGlobalCoord(int i) override {
    assert(0 <= i && i < CountPoints());
    return global_coords_[i];
  }
  Scalar &GetGlobalWeight(int i) override {
    assert(0 <= i && i < CountPoints());
    return global_weights_[i];
  }

 public:
  explicit Hexahedron(Lagrange const &lagrange)
      : lagrange_(&lagrange) {
    volume_ = this->BuildQuadraturePoints();
  }
  Hexahedron(const Hexahedron &) = default;
  Hexahedron &operator=(const Hexahedron &) = default;
  Hexahedron(Hexahedron &&) noexcept = default;
  Hexahedron &operator=(Hexahedron &&) noexcept = default;
  virtual ~Hexahedron() noexcept = default;

  const Lagrange &lagrange() const override {
    return *lagrange_;
  }

  Scalar volume() const override {
    return volume_;
  }
};

template <std::floating_point Scalar, int Qx, int Qy, int Qz>
std::array<typename Hexahedron<Scalar, Qx, Qy, Qz>::Local, Qx * Qy * Qz> const
Hexahedron<Scalar, Qx, Qy, Qz>::local_coords_
    = Hexahedron<Scalar, Qx, Qy, Qz>::BuildLocalCoords();

template <std::floating_point Scalar, int Qx, int Qy, int Qz>
std::array<Scalar, Qx * Qy * Qz> const
Hexahedron<Scalar, Qx, Qy, Qz>::local_weights_
    = Hexahedron<Scalar, Qx, Qy, Qz>::BuildLocalWeights();

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_HEXAHEDRON_HPP_
