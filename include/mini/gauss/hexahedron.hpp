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
  using Mat3x3 = algebra::Matrix<Scalar, 3, 3>;
  using Mat1x8 = algebra::Matrix<Scalar, 1, 8>;
  using Mat8x1 = algebra::Matrix<Scalar, 8, 1>;
  using Mat3x8 = algebra::Matrix<Scalar, 3, 8>;
  using Mat8x3 = algebra::Matrix<Scalar, 8, 3>;
  using Mat3x1 = algebra::Matrix<Scalar, 3, 1>;

  using Arr1x8 = algebra::Array<Scalar, 1, 8>;
  using Arr8x1 = algebra::Array<Scalar, 8, 1>;
  using Arr3x8 = algebra::Array<Scalar, 3, 8>;
  using Arr8x3 = algebra::Array<Scalar, 8, 3>;

  using GaussX = GaussLegendre<Scalar, Qx>;
  using GaussY = GaussLegendre<Scalar, Qy>;
  using GaussZ = GaussLegendre<Scalar, Qz>;

  using Base = Cell<Scalar>;
  using Lagrange = lagrange::Hexahedron<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::LocalCoord;
  using typename Base::GlobalCoord;
  using typename Base::Jacobian;

 private:
  static const std::array<LocalCoord, Qx * Qy * Qz> local_coords_;
  static const std::array<Scalar, Qx * Qy * Qz> local_weights_;
  std::array<GlobalCoord, Qx * Qy * Qz> global_coords_;
  std::array<Scalar, Qx * Qy * Qz> global_weights_;
  Lagrange const *lagrange_;
  Scalar volume_;

 public:
  int CountQuadraturePoints() const override {
    return Qx * Qy * Qz;
  }
  template <typename T, typename U>
  static void SortNodesOnFace(const T *cell_nodes, U *face_nodes) {
    Lagrange::SortNodesOnFace(cell_nodes, face_nodes);
  }

 private:
  void BuildQuadraturePoints() {
    int n = CountQuadraturePoints();
    volume_ = 0.0;
    for (int i = 0; i < n; ++i) {
      const Base *const_this = this;
      auto det_j = const_this->LocalToJacobian(GetLocalCoord(i)).determinant();
      global_weights_[i] = local_weights_[i] * det_j;
      volume_ += global_weights_[i];
      global_coords_[i] = const_this->LocalToGlobal(GetLocalCoord(i));
    }
  }
  static constexpr auto BuildLocalCoords() {
    std::array<LocalCoord, Qx * Qy * Qz> points;
    int n = 0;
    for (int i = 0; i < Qx; ++i) {
      for (int j = 0; j < Qy; ++j) {
        for (int k = 0; k < Qz; ++k) {
          points[n][0] = GaussX::points[i];
          points[n][1] = GaussY::points[j];
          points[n][2] = GaussZ::points[k];
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

 public:
  explicit Hexahedron(Lagrange const &lagrange)
      : lagrange_(&lagrange) {
    BuildQuadraturePoints();
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
std::array<typename Hexahedron<Scalar, Qx, Qy, Qz>::LocalCoord, Qx * Qy * Qz> const
Hexahedron<Scalar, Qx, Qy, Qz>::local_coords_
    = Hexahedron<Scalar, Qx, Qy, Qz>::BuildLocalCoords();

template <std::floating_point Scalar, int Qx, int Qy, int Qz>
std::array<Scalar, Qx * Qy * Qz> const
Hexahedron<Scalar, Qx, Qy, Qz>::local_weights_
    = Hexahedron<Scalar, Qx, Qy, Qz>::BuildLocalWeights();

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_HEXAHEDRON_HPP_
