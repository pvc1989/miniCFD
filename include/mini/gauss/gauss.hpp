//  Copyright 2019 PEI Weicheng and YANG Minghao
#ifndef MINI_GAUSS_GAUSS_HPP_
#define MINI_GAUSS_GAUSS_HPP_

#include <array>
#include <cmath>

namespace mini {
namespace gauss {

template <typename Scalar = double, int Q = 4>
struct GaussLegendre;

template <typename Scalar>
struct GaussLegendre<Scalar, 1> {
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
template <typename Scalar>
typename GaussLegendre<Scalar, 1>::Array const
GaussLegendre<Scalar, 1>::points =
    GaussLegendre<Scalar, 1>::BuildPoints();
template <typename Scalar>
typename GaussLegendre<Scalar, 1>::Array const
GaussLegendre<Scalar, 1>::weights =
    GaussLegendre<Scalar, 1>::BuildWeights();

template <typename Scalar>
struct GaussLegendre<Scalar, 2> {
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
template <typename Scalar>
typename GaussLegendre<Scalar, 2>::Array const
GaussLegendre<Scalar, 2>::points =
    GaussLegendre<Scalar, 2>::BuildPoints();
template <typename Scalar>
typename GaussLegendre<Scalar, 2>::Array const
GaussLegendre<Scalar, 2>::weights =
    GaussLegendre<Scalar, 2>::BuildWeights();

template <typename Scalar>
struct GaussLegendre<Scalar, 3> {
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
template <typename Scalar>
typename GaussLegendre<Scalar, 3>::Array const
GaussLegendre<Scalar, 3>::points =
    GaussLegendre<Scalar, 3>::BuildPoints();
template <typename Scalar>
typename GaussLegendre<Scalar, 3>::Array const
GaussLegendre<Scalar, 3>::weights =
    GaussLegendre<Scalar, 3>::BuildWeights();

template <typename Scalar>
struct GaussLegendre<Scalar, 4> {
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
template <typename Scalar>
typename GaussLegendre<Scalar, 4>::Array const
GaussLegendre<Scalar, 4>::points =
    GaussLegendre<Scalar, 4>::BuildPoints();
template <typename Scalar>
typename GaussLegendre<Scalar, 4>::Array const
GaussLegendre<Scalar, 4>::weights =
    GaussLegendre<Scalar, 4>::BuildWeights();

template <typename Scalar>
struct GaussLegendre<Scalar, 5> {
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
template <typename Scalar>
typename GaussLegendre<Scalar, 5>::Array const
GaussLegendre<Scalar, 5>::points =
    GaussLegendre<Scalar, 5>::BuildPoints();
template <typename Scalar>
typename GaussLegendre<Scalar, 5>::Array const
GaussLegendre<Scalar, 5>::weights =
    GaussLegendre<Scalar, 5>::BuildWeights();

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_GAUSS_HPP_
