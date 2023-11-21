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
 * @tparam S  Type of scalar variables.
 * @tparam Q  Nnumber of quadrature points.
 */
template <std::floating_point S = double, int Q = 4>
struct Legendre;

template <std::floating_point S>
struct Legendre<S, 1> {
  using Array = std::array<S, 1>;
  using Scalar = S;
  static constexpr int Q = 1;
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

template <std::floating_point S>
struct Legendre<S, 2> {
  using Array = std::array<S, 2>;
  using Scalar = S;
  static constexpr int Q = 2;
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

template <std::floating_point S>
struct Legendre<S, 3> {
  using Array = std::array<S, 3>;
  using Scalar = S;
  static constexpr int Q = 3;
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

template <std::floating_point S>
struct Legendre<S, 4> {
  using Array = std::array<S, 4>;
  using Scalar = S;
  static constexpr int Q = 4;
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

template <std::floating_point S>
struct Legendre<S, 5> {
  using Array = std::array<S, 5>;
  using Scalar = S;
  static constexpr int Q = 5;
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

template <std::floating_point S>
struct Legendre<S, 6> {
  using Array = std::array<S, 6>;
  using Scalar = S;
  static constexpr int Q = 6;
  static const Array points;
  static const Array weights;
  static Array BuildPoints() {
    return {
        -0.932469514203152,
        -0.6612093864662645,
        -0.23861918608319693,
        +0.23861918608319693,
        +0.6612093864662645,
        +0.932469514203152,
    };
  }
  static Array BuildWeights() {
    return {
        0.17132449237917016,
        0.36076157304813855,
        0.4679139345726912,
        0.4679139345726912,
        0.36076157304813855,
        0.17132449237917016,
    };
  }
};
template <std::floating_point Scalar>
typename Legendre<Scalar, 6>::Array const
Legendre<Scalar, 6>::points =
    Legendre<Scalar, 6>::BuildPoints();
template <std::floating_point Scalar>
typename Legendre<Scalar, 6>::Array const
Legendre<Scalar, 6>::weights =
    Legendre<Scalar, 6>::BuildWeights();

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_LEGENDRE_HPP_
