//  Copyright 2019 PEI Weicheng and YANG Minghao
#ifndef MINI_GAUSS_LEGENDRE_HPP_
#define MINI_GAUSS_LEGENDRE_HPP_

#include <concepts>

#include <cmath>

#include <array>

namespace mini {
namespace gauss {

/**
 * @brief Gauss--Legendre quadrature rules, i.e. \f$ \int_{-1}^{1} f(\xi) \,\mathrm{d}\xi \approx \sum_{q=1}^{Q} w_q f(\xi_q) \f$
 * 
 * @tparam Scalar  Type of scalar variables.
 * @tparam Q  Nnumber of quadrature points.
 */
template <std::floating_point Scalar = double, int Q = 4>
struct Legendre;

template <std::floating_point Scalar>
struct Legendre<Scalar, 1> {
  using Array = std::array<Scalar, 1>;
  static const Array points;
  static const Array weights;
  static Array BuildPoints() {
    return { 0.0 };
  }
  static Array BuildWeights() {
    return { 2.0 };
  }
};
template <std::floating_point Scalar>
typename Legendre<Scalar, 1>::Array const
Legendre<Scalar, 1>::points =
    Legendre<Scalar, 1>::BuildPoints();
template <std::floating_point Scalar>
typename Legendre<Scalar, 1>::Array const
Legendre<Scalar, 1>::weights =
    Legendre<Scalar, 1>::BuildWeights();

template <std::floating_point Scalar>
struct Legendre<Scalar, 2> {
  using Array = std::array<Scalar, 2>;
  static const Array points;
  static const Array weights;
  static Array BuildPoints() {
    return { -std::sqrt(Scalar(1.0/3.0)), +std::sqrt(Scalar(1.0/3.0)) };
  }
  static Array BuildWeights() {
    return { 1.0, 1.0 };
  }
};
template <std::floating_point Scalar>
typename Legendre<Scalar, 2>::Array const
Legendre<Scalar, 2>::points =
    Legendre<Scalar, 2>::BuildPoints();
template <std::floating_point Scalar>
typename Legendre<Scalar, 2>::Array const
Legendre<Scalar, 2>::weights =
    Legendre<Scalar, 2>::BuildWeights();

template <std::floating_point Scalar>
struct Legendre<Scalar, 3> {
  using Array = std::array<Scalar, 3>;
  static const Array points;
  static const Array weights;
  static Array BuildPoints() {
    return { -std::sqrt(Scalar(0.6)), 0.0, +std::sqrt(Scalar(0.6)) };
  }
  static Array BuildWeights() {
    return { 5.0/9.0, 8.0/9.0, 5.0/9.0 };
  }
};
template <std::floating_point Scalar>
typename Legendre<Scalar, 3>::Array const
Legendre<Scalar, 3>::points =
    Legendre<Scalar, 3>::BuildPoints();
template <std::floating_point Scalar>
typename Legendre<Scalar, 3>::Array const
Legendre<Scalar, 3>::weights =
    Legendre<Scalar, 3>::BuildWeights();

template <std::floating_point Scalar>
struct Legendre<Scalar, 4> {
  using Array = std::array<Scalar, 4>;
  static const Array points;
  static const Array weights;
  static Array BuildPoints() {
    return {
        -std::sqrt((3 + 2 * std::sqrt(Scalar(1.2))) / 7),
        -std::sqrt((3 - 2 * std::sqrt(Scalar(1.2))) / 7),
        +std::sqrt((3 - 2 * std::sqrt(Scalar(1.2))) / 7),
        +std::sqrt((3 + 2 * std::sqrt(Scalar(1.2))) / 7),
    };
  }
  static Array BuildWeights() {
    return {
        (18 - std::sqrt(Scalar(30.0))) / 36,
        (18 + std::sqrt(Scalar(30.0))) / 36,
        (18 + std::sqrt(Scalar(30.0))) / 36,
        (18 - std::sqrt(Scalar(30.0))) / 36,
    };
  }
};
template <std::floating_point Scalar>
typename Legendre<Scalar, 4>::Array const
Legendre<Scalar, 4>::points =
    Legendre<Scalar, 4>::BuildPoints();
template <std::floating_point Scalar>
typename Legendre<Scalar, 4>::Array const
Legendre<Scalar, 4>::weights =
    Legendre<Scalar, 4>::BuildWeights();

template <std::floating_point Scalar>
struct Legendre<Scalar, 5> {
  using Array = std::array<Scalar, 5>;
  static const Array points;
  static const Array weights;
  static Array BuildPoints() {
    return {
        -std::sqrt((5 + std::sqrt(Scalar(40 / 7.0))) / 9),
        -std::sqrt((5 - std::sqrt(Scalar(40 / 7.0))) / 9),
        0,
        +std::sqrt((5 - std::sqrt(Scalar(40 / 7.0))) / 9),
        +std::sqrt((5 + std::sqrt(Scalar(40 / 7.0))) / 9),
    };
  }
  static Array BuildWeights() {
    return {
        (322 - 13 * std::sqrt(Scalar(70.0))) / 900,
        (322 + 13 * std::sqrt(Scalar(70.0))) / 900,
        128.0 / 225.0,
        (322 + 13 * std::sqrt(Scalar(70.0))) / 900,
        (322 - 13 * std::sqrt(Scalar(70.0))) / 900,
    };
  }
};
template <std::floating_point Scalar>
typename Legendre<Scalar, 5>::Array const
Legendre<Scalar, 5>::points =
    Legendre<Scalar, 5>::BuildPoints();
template <std::floating_point Scalar>
typename Legendre<Scalar, 5>::Array const
Legendre<Scalar, 5>::weights =
    Legendre<Scalar, 5>::BuildWeights();

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_LEGENDRE_HPP_
