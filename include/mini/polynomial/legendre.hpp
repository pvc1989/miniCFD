//  Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_POLYNOMIAL_LEGENDRE_HPP_
#define MINI_POLYNOMIAL_LEGENDRE_HPP_

#include "mini/algebra/column.hpp"

namespace mini {
namespace polynomial {

template <int kDof>
using Vector = algebra::Column<double, kDof>;

template <int kDegree>
constexpr double legendre(double x) {
  constexpr auto a_prev = (2.0 * kDegree - 1) / kDegree;
  constexpr auto a_prev_prev = (1.0 - kDegree) / kDegree;
  return legendre<kDegree-1>(x) * a_prev * x
       + legendre<kDegree-2>(x) * a_prev_prev;
}
template <>
constexpr double legendre<0>(double x) { return 1.0; }
template <>
constexpr double legendre<1>(double x) { return x; }

template <int kDegree>
constexpr void legendre_recursive(double x, double* result) {
  constexpr auto a_prev = (2.0 * kDegree - 1) / kDegree;
  constexpr auto a_prev_prev = (1.0 - kDegree) / kDegree;
  legendre_recursive<kDegree-1>(x, result-1);
  *result = *(result-1) * a_prev * x 
          + *(result-2) * a_prev_prev;
}
template <>
constexpr void legendre_recursive<0>(double x, double* result) {
  *result = 1.0;
}
template <>
constexpr void legendre_recursive<1>(double x, double* result) {
  *result = x;
}

template <int kDegree>
constexpr Vector<kDegree+1> legendre_array(double x) {
  Vector<kDegree+1> result;
  result[0] = 1.0;
  legendre_recursive<kDegree>(x, &result[kDegree]);
  return result;
}

}  // namespace polynomial
}  // namespace mini

#endif  // MINI_POLYNOMIAL_LEGENDRE_HPP_
