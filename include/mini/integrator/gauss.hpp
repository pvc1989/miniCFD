//  Copyright 2019 PEI Weicheng and YANG Minghao
#ifndef MINI_INTEGRATOR_GAUSS_HPP_
#define MINI_INTEGRATOR_GAUSS_HPP_

#include <array>
#include <cmath>

#include "mini/algebra/eigen.hpp"

namespace mini {
namespace integrator {

template <typename Scalar = double, int Q = 4>
struct GaussLegendre;

template <typename Scalar>
struct GaussLegendre<Scalar, 1> {
  using Mat1x1 = algebra::Matrix<Scalar, 1, 1>;
  static const Mat1x1 points;
  static const Mat1x1 weights;
  static Mat1x1 BuildPoints() {
    return { 0.0 };
  }
  static Mat1x1 BuildWeights() {
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
  using Mat1x2 = algebra::Matrix<Scalar, 1, 2>;
  static const Mat1x2 points;
  static const Mat1x2 weights;
  static Mat1x2 BuildPoints() {
    return { -std::sqrt(1.0/3.0), +std::sqrt(1.0/3.0) };
  }
  static Mat1x2 BuildWeights() {
    return { 1.0, 1.0 };
  }
};
template <typename Scalar>
typename GaussLegendre<Scalar, 2>::Mat1x2 const
GaussLegendre<Scalar, 2>::points =
    GaussLegendre<Scalar, 2>::BuildPoints();
template <typename Scalar>
typename GaussLegendre<Scalar, 2>::Mat1x2 const
GaussLegendre<Scalar, 2>::weights =
    GaussLegendre<Scalar, 2>::BuildWeights();

template <typename Scalar>
struct GaussLegendre<Scalar, 3> {
  using Mat1x3 = algebra::Matrix<Scalar, 1, 3>;
  static const Mat1x3 points;
  static const Mat1x3 weights;
  static Mat1x3 BuildPoints() {
    return { -std::sqrt(0.6), 0.0, +std::sqrt(0.6) };
  }
  static Mat1x3 BuildWeights() {
    return { 5.0/9.0, 8.0/9.0, 5.0/9.0 };
  }
};
template <typename Scalar>
typename GaussLegendre<Scalar, 3>::Mat1x3 const
GaussLegendre<Scalar, 3>::points =
    GaussLegendre<Scalar, 3>::BuildPoints();
template <typename Scalar>
typename GaussLegendre<Scalar, 3>::Mat1x3 const
GaussLegendre<Scalar, 3>::weights =
    GaussLegendre<Scalar, 3>::BuildWeights();

template <typename Scalar>
struct GaussLegendre<Scalar, 4> {
  using Mat1x4 = algebra::Matrix<Scalar, 1, 4>;
  static const Mat1x4 points;
  static const Mat1x4 weights;
  static Mat1x4 BuildPoints() {
    return {
        -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7),
        -std::sqrt((3 - 2 * std::sqrt(1.2)) / 7),
        +std::sqrt((3 - 2 * std::sqrt(1.2)) / 7),
        +std::sqrt((3 + 2 * std::sqrt(1.2)) / 7),
    };
  }
  static Mat1x4 BuildWeights() {
    return {
        (18 - std::sqrt(30)) / 36,
        (18 + std::sqrt(30)) / 36,
        (18 + std::sqrt(30)) / 36,
        (18 - std::sqrt(30)) / 36,
    };
  }
};
template <typename Scalar>
typename GaussLegendre<Scalar, 4>::Mat1x4 const
GaussLegendre<Scalar, 4>::points =
    GaussLegendre<Scalar, 4>::BuildPoints();
template <typename Scalar>
typename GaussLegendre<Scalar, 4>::Mat1x4 const
GaussLegendre<Scalar, 4>::weights =
    GaussLegendre<Scalar, 4>::BuildWeights();

template <typename Scalar>
struct GaussLegendre<Scalar, 5> {
  using Mat1x5 = algebra::Matrix<Scalar, 1, 5>;
  static const Mat1x5 points;
  static const Mat1x5 weights;
  static Mat1x5 BuildPoints() {
    return {
        -std::sqrt((5 + std::sqrt(40 / 7.0)) / 9),
        -std::sqrt((5 - std::sqrt(40 / 7.0)) / 9),
        0,
        +std::sqrt((5 - std::sqrt(40 / 7.0)) / 9),
        +std::sqrt((5 + std::sqrt(40 / 7.0)) / 9),
    };
  }
  static Mat1x5 BuildWeights() {
    return {
        (322 - 13 * std::sqrt(70.0)) / 900,
        (322 + 13 * std::sqrt(70.0)) / 900,
        128.0 / 225.0,
        (322 + 13 * std::sqrt(70.0)) / 900,
        (322 - 13 * std::sqrt(70.0)) / 900,
    };
  }
};
template <typename Scalar>
typename GaussLegendre<Scalar, 5>::Mat1x5 const
GaussLegendre<Scalar, 5>::points =
    GaussLegendre<Scalar, 5>::BuildPoints();
template <typename Scalar>
typename GaussLegendre<Scalar, 5>::Mat1x5 const
GaussLegendre<Scalar, 5>::weights =
    GaussLegendre<Scalar, 5>::BuildWeights();

}  // namespace integrator
}  // namespace mini

#endif  // MINI_INTEGRATOR_GAUSS_HPP_
