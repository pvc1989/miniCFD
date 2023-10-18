//  Copyright 2023 PEI Weicheng
#ifndef MINI_GAUSS_LOBATTO_HPP_
#define MINI_GAUSS_LOBATTO_HPP_

#include <concepts>

#include <cmath>

#include <array>

namespace mini {
namespace gauss {

/**
 * @brief Gauss--Lobatto quadrature rules, i.e. \f$ \int_{-1}^{1} f(\xi) \,\mathrm{d}\xi \approx \sum_{q=1}^{Q} w_q f(\xi_q) \f$, in which \f$ \xi_1 = -1 \f$ and \f$ \xi_Q = +1 \f$.
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam Q  Nnumber of quadrature points.
 */
template <std::floating_point Scalar = double, int Q = 4>
struct Lobatto;

template <std::floating_point Scalar>
struct Lobatto<Scalar, 2> {
  using Array = std::array<Scalar, 2>;
  static const Array points;
  static const Array weights;
  static Array BuildPoints() {
    return { -1.0, +1.0 };
  }
  static Array BuildWeights() {
    return { 1.0, 1.0 };
  }
};
template <std::floating_point Scalar>
typename Lobatto<Scalar, 2>::Array const
Lobatto<Scalar, 2>::points =
    Lobatto<Scalar, 2>::BuildPoints();
template <std::floating_point Scalar>
typename Lobatto<Scalar, 2>::Array const
Lobatto<Scalar, 2>::weights =
    Lobatto<Scalar, 2>::BuildWeights();

template <std::floating_point Scalar>
struct Lobatto<Scalar, 3> {
  using Array = std::array<Scalar, 3>;
  static const Array points;
  static const Array weights;
  static Array BuildPoints() {
    return { -1.0, 0.0, +1.0 };
  }
  static Array BuildWeights() {
    return { 1.0/3, 4.0/3, 1.0/3 };
  }
};
template <std::floating_point Scalar>
typename Lobatto<Scalar, 3>::Array const
Lobatto<Scalar, 3>::points =
    Lobatto<Scalar, 3>::BuildPoints();
template <std::floating_point Scalar>
typename Lobatto<Scalar, 3>::Array const
Lobatto<Scalar, 3>::weights =
    Lobatto<Scalar, 3>::BuildWeights();

template <std::floating_point Scalar>
struct Lobatto<Scalar, 4> {
  using Array = std::array<Scalar, 4>;
  static const Array points;
  static const Array weights;
  static Array BuildPoints() {
    return { -1.0, -std::sqrt(1.0/5), +std::sqrt(1.0/5), +1.0 };
  }
  static Array BuildWeights() {
    return { 1.0/6, 5.0/6, 5.0/6, 1.0/6 };
  }
};
template <std::floating_point Scalar>
typename Lobatto<Scalar, 4>::Array const
Lobatto<Scalar, 4>::points =
    Lobatto<Scalar, 4>::BuildPoints();
template <std::floating_point Scalar>
typename Lobatto<Scalar, 4>::Array const
Lobatto<Scalar, 4>::weights =
    Lobatto<Scalar, 4>::BuildWeights();

template <std::floating_point Scalar>
struct Lobatto<Scalar, 5> {
  using Array = std::array<Scalar, 5>;
  static const Array points;
  static const Array weights;
  static Array BuildPoints() {
    return { -1.0, -std::sqrt(21.0)/7, 0.0, +std::sqrt(21.0)/7, +1.0 };
  }
  static Array BuildWeights() {
    return { 0.1, 49.0/90, 32.0/45, 49.0/90, 0.1 };
  }
};
template <std::floating_point Scalar>
typename Lobatto<Scalar, 5>::Array const
Lobatto<Scalar, 5>::points =
    Lobatto<Scalar, 5>::BuildPoints();
template <std::floating_point Scalar>
typename Lobatto<Scalar, 5>::Array const
Lobatto<Scalar, 5>::weights =
    Lobatto<Scalar, 5>::BuildWeights();

template <std::floating_point Scalar>
struct Lobatto<Scalar, 6> {
  using Array = std::array<Scalar, 6>;
  static const Array points;
  static const Array weights;
  static Array BuildPoints() {
    return {
        -1.0,
        -std::sqrt((7 + 2 * std::sqrt(7.0)) / 21),
        -std::sqrt((7 - 2 * std::sqrt(7.0)) / 21),
        +std::sqrt((7 - 2 * std::sqrt(7.0)) / 21),
        +std::sqrt((7 + 2 * std::sqrt(7.0)) / 21),
        +1.0
    };
  }
  static Array BuildWeights() {
    return {
        1.0 / 15,
        (14.0 - std::sqrt(7.0)) / 30,
        (14.0 + std::sqrt(7.0)) / 30,
        (14.0 + std::sqrt(7.0)) / 30,
        (14.0 - std::sqrt(7.0)) / 30,
        1.0 / 15
    };
  }
};
template <std::floating_point Scalar>
typename Lobatto<Scalar, 6>::Array const
Lobatto<Scalar, 6>::points =
    Lobatto<Scalar, 6>::BuildPoints();
template <std::floating_point Scalar>
typename Lobatto<Scalar, 6>::Array const
Lobatto<Scalar, 6>::weights =
    Lobatto<Scalar, 6>::BuildWeights();

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_LOBATTO_HPP_
