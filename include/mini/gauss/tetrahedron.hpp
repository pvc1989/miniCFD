//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_GAUSS_TETRAHEDRON_HPP_
#define MINI_GAUSS_TETRAHEDRON_HPP_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstring>
#include <type_traits>
#include <utility>

#include "mini/gauss/cell.hpp"
#include "mini/lagrange/tetrahedron.hpp"

namespace mini {
namespace gauss {

/**
 * @brief Numerical integrators on tetrahedral elements.
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam kPoints  Number of qudrature points.
 */
template <std::floating_point Scalar, int kPoints>
class Tetrahedron : public Cell<Scalar> {
 public:
  using Lagrange = lagrange::Tetrahedron<Scalar>;
  using Real = typename Lagrange::Real;
  using Local = typename Lagrange::Local;
  using Global = typename Lagrange::Global;
  using Jacobian = typename Lagrange::Jacobian;

 private:
  static const std::array<Local, kPoints> local_coords_;
  static const std::array<Scalar, kPoints> local_weights_;
  std::array<Global, kPoints> global_coords_;
  std::array<Scalar, kPoints> global_weights_;
  Lagrange const *lagrange_;
  Scalar volume_;

 public:
  int CountQuadraturePoints() const override {
    return kPoints;
  }

 public:
  const Global &GetGlobalCoord(int i) const override {
    return global_coords_[i];
  }
  const Scalar &GetGlobalWeight(int i) const override {
    return global_weights_[i];
  }
  const Local &GetLocalCoord(int i) const override {
    return local_coords_[i];
  }
  const Scalar &GetLocalWeight(int i) const override {
    return local_weights_[i];
  }

 protected:
  Global &GetGlobalCoord(int i) override {
    return global_coords_[i];
  }
  Scalar &GetGlobalWeight(int i) override {
    return global_weights_[i];
  }

 public:
  explicit Tetrahedron(Lagrange const &lagrange)
      : lagrange_(&lagrange) {
    volume_ = this->BuildQuadraturePoints();
  }
  Tetrahedron(const Tetrahedron &) = default;
  Tetrahedron &operator=(const Tetrahedron &) = default;
  Tetrahedron(Tetrahedron &&) noexcept = default;
  Tetrahedron &operator=(Tetrahedron &&) noexcept = default;
  virtual ~Tetrahedron() noexcept = default;

  const Lagrange &lagrange() const override {
    return *lagrange_;
  }

  Scalar volume() const override {
    return volume_;
  }
};

template <std::floating_point Scalar, int kPoints>
class TetrahedronBuilder;

template <std::floating_point Scalar, int kPoints>
const std::array<typename Tetrahedron<Scalar, kPoints>::Local, kPoints>
Tetrahedron<Scalar, kPoints>::local_coords_
    = TetrahedronBuilder<Scalar, kPoints>::BuildLocalCoords();

template <std::floating_point Scalar, int kPoints>
const std::array<Scalar, kPoints>
Tetrahedron<Scalar, kPoints>::local_weights_
    = TetrahedronBuilder<Scalar, kPoints>::BuildLocalWeights();

template <std::floating_point Scalar>
class TetrahedronBuilder<Scalar, 1> {
  static constexpr int kPoints = 1;
  using Local = typename Tetrahedron<Scalar, kPoints>::Local;

 public:
  static constexpr auto BuildLocalCoords() {
    Scalar a = 0.25;
    std::array<Local, kPoints> points;
    points[0] = { a, a, a };
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    std::array<Scalar, kPoints> weights{ 1.0 / 6.0 };
    return weights;
  }
};

template <std::floating_point Scalar>
class TetrahedronBuilder<Scalar, 4> {
  static constexpr int kPoints = 4;
  using Local = typename Tetrahedron<Scalar, kPoints>::Local;

 public:
  static constexpr auto BuildLocalCoords() {
    std::array<Local, kPoints> points;
    int q = 0;
    // the only S31 orbit
    Scalar a = 0.13819660112501051517954131656343619;
    auto c = 1 - 3 * a;
    points[q++] = { a, a, a };
    points[q++] = { a, a, c };
    points[q++] = { a, c, a };
    points[q++] = { c, a, a };
    assert(q == kPoints);
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    std::array<Scalar, kPoints> weights;
    for (int q = 0; q < kPoints; ++q)
      weights[q] = 0.25 / 6.0;
    return weights;
  }
};

template <std::floating_point Scalar>
class TetrahedronBuilder<Scalar, 14> {
  static constexpr int kPoints = 14;
  using Local = typename Tetrahedron<Scalar, kPoints>::Local;

 public:
  static constexpr auto BuildLocalCoords() {
    std::array<Local, kPoints> points;
    int q = 0;
    // the two S31 orbits
    Scalar a_s31[] = {
        0.31088591926330060979734573376345783,
        0.09273525031089122640232391373703061 };
    for (auto a : a_s31) {
      auto c = 1 - 3 * a;
      points[q++] = { a, a, a };
      points[q++] = { a, a, c };
      points[q++] = { a, c, a };
      points[q++] = { c, a, a };
    }
    {  // the only S22 orbit
      Scalar a = 0.04550370412564964949188052627933943;
      auto c = (1 - 2 * a) * .5;
      points[q++] = { a, a, c };
      points[q++] = { a, c, a };
      points[q++] = { a, c, c };
      points[q++] = { c, a, a };
      points[q++] = { c, a, c };
      points[q++] = { c, c, a };
    }
    assert(q == kPoints);
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    std::array<Scalar, kPoints> weights;
    for (int q = 0; q < 4; ++q)
      weights[q] = 0.11268792571801585079918565233328633;
    for (int q = 4; q < 8; ++q)
      weights[q] = 0.07349304311636194954371020548632750;
    for (int q = 8; q < kPoints; ++q)
      weights[q] = 0.04254602077708146643806942812025744;
    for (int q = 0; q < kPoints; ++q)
      weights[q] /= 6.0;
    return weights;
  }
};

template <std::floating_point Scalar>
class TetrahedronBuilder<Scalar, 15> {
  static constexpr int kPoints = 15;
  using Local = typename Tetrahedron<Scalar, kPoints>::Local;

 public:
  static constexpr auto BuildLocalCoords() {
    std::array<Local, kPoints> points;
    int q = 0;
    {  // the only S4 orbit
      Scalar a = 0.25;
      points[q++] = { a, a, a };
    }
    // the two S31 orbits
    Scalar a_s31[] = {
        0.09197107805272303,
        0.31979362782962991 };
    for (auto a : a_s31) {
      auto c = 1 - 3 * a;
      points[q++] = { a, a, a };
      points[q++] = { a, a, c };
      points[q++] = { a, c, a };
      points[q++] = { c, a, a };
    }
    {  // the only S22 orbit
      Scalar a = 0.05635083268962916;
      auto c = (1 - 2 * a) * .5;
      points[q++] = { a, a, c };
      points[q++] = { a, c, a };
      points[q++] = { a, c, c };
      points[q++] = { c, a, a };
      points[q++] = { c, a, c };
      points[q++] = { c, c, a };
    }
    assert(q == kPoints);
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    std::array<Scalar, kPoints> weights;
    for (int q = 0; q < 1; ++q)
      weights[q] = 16.0 / 135.0;
    for (int q = 1; q < 5; ++q)
      weights[q] = 0.07193708377901862;
    for (int q = 5; q < 9; ++q)
      weights[q] = 0.06906820722627239;
    for (int q = 9; q < kPoints; ++q)
      weights[q] = 20.0 / 378.0;
    for (int q = 0; q < kPoints; ++q)
      weights[q] /= 6.0;
    return weights;
  }
};

template <std::floating_point Scalar>
class TetrahedronBuilder<Scalar, 24> {
  static constexpr int kPoints = 24;
  using Local = typename Tetrahedron<Scalar, kPoints>::Local;

 public:
  static constexpr auto BuildLocalCoords() {
    std::array<Local, kPoints> points;
    int q = 0;
    // the three S31 orbits
    Scalar a_s31[] = {
        0.21460287125915202928883921938628499,
        0.04067395853461135311557944895641006,
        0.32233789014227551034399447076249213 };
    for (auto a : a_s31) {
      auto c = 1 - 3 * a;
      points[q++] = { a, a, a };
      points[q++] = { a, a, c };
      points[q++] = { a, c, a };
      points[q++] = { c, a, a };
    }
    {  // the only S211 orbit
      Scalar a = 0.06366100187501752529923552760572698;
      Scalar b = 0.60300566479164914136743113906093969;
      auto c = 1 - a - a - b;
      points[q++] = { a, a, b };
      points[q++] = { a, a, c };
      points[q++] = { a, b, a };
      points[q++] = { a, b, c };
      points[q++] = { a, c, a };
      points[q++] = { a, c, b };
      points[q++] = { b, a, a };
      points[q++] = { b, a, c };
      points[q++] = { b, c, a };
      points[q++] = { c, a, a };
      points[q++] = { c, a, b };
      points[q++] = { c, b, a };
    }
    assert(q == kPoints);
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    std::array<Scalar, kPoints> weights;
    for (int q = 0; q < 4; ++q)
      weights[q] = 0.03992275025816749209969062755747998;
    for (int q = 4; q < 8; ++q)
      weights[q] = 0.01007721105532064294801323744593686;
    for (int q = 8; q < 12; ++q)
      weights[q] = 0.05535718154365472209515327785372602;
    for (int q = 12; q < 24; ++q)
      weights[q] = 27./560.;
    for (int q = 0; q < 24; ++q)
      weights[q] /= 6.0;
    return weights;
  }
};

template <std::floating_point Scalar>
class TetrahedronBuilder<Scalar, 46> {
  static constexpr int kPoints = 46;
  using Local = typename Tetrahedron<Scalar, kPoints>::Local;

 public:
  static constexpr auto BuildLocalCoords() {
    std::array<Local, kPoints> points;
    int q = 0;
    // the four S31 orbits
    Scalar a_s31[] = {
        .03967542307038990126507132953938949,
        .31448780069809631378416056269714830,
        .10198669306270330000000000000000000,
        .18420369694919151227594641734890918 };
    for (auto a : a_s31) {
      auto c = 1 - 3 * a;
      points[q++] = { a, a, a };
      points[q++] = { a, a, c };
      points[q++] = { a, c, a };
      points[q++] = { c, a, a };
    }
    {  // the only S22 orbit
      Scalar a = .06343628775453989240514123870189827;
      auto c = (1 - 2 * a) * .5;
      points[q++] = { a, a, c };
      points[q++] = { a, c, a };
      points[q++] = { a, c, c };
      points[q++] = { c, a, a };
      points[q++] = { c, a, c };
      points[q++] = { c, c, a };
    }
    // the two S211 orbits
    std::pair<Scalar, Scalar> ab_s211[2]{
      { .02169016206772800480266248262493018,
        .71993192203946593588943495335273478 },
      { .20448008063679571424133557487274534,
        .58057719012880922417539817139062041 }};
    for (auto [a, b] : ab_s211) {
      auto c = 1 - a - a - b;
      points[q++] = { a, a, b };
      points[q++] = { a, a, c };
      points[q++] = { a, b, a };
      points[q++] = { a, b, c };
      points[q++] = { a, c, a };
      points[q++] = { a, c, b };
      points[q++] = { b, a, a };
      points[q++] = { b, a, c };
      points[q++] = { b, c, a };
      points[q++] = { c, a, a };
      points[q++] = { c, a, b };
      points[q++] = { c, b, a };
    }
    assert(q == kPoints);
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    std::array<Scalar, kPoints> weights;
    for (int q = 0; q < 4; ++q)
      weights[q] = .00639714777990232132145142033517302;
    for (int q = 4; q < 8; ++q)
      weights[q] = .04019044802096617248816115847981783;
    for (int q = 8; q < 12; ++q)
      weights[q] = .02430797550477032117486910877192260;
    for (int q = 12; q < 16; ++q)
      weights[q] = .05485889241369744046692412399039144;
    for (int q = 16; q < 22; ++q)
      weights[q] = .03571961223409918246495096899661762;
    for (int q = 22; q < 34; ++q)
      weights[q] = .00718319069785253940945110521980376;
    for (int q = 34; q < 46; ++q)
      weights[q] = .01637218194531911754093813975611913;
    for (int q = 0; q < 46; ++q)
      weights[q] /= 6.0;
    return weights;
  }
};

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_TETRAHEDRON_HPP_
