//  Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_POLYNOMIAL_DERIVATIVE_HPP_
#define MINI_POLYNOMIAL_DERIVATIVE_HPP_

#include "mini/algebra/column.hpp"

namespace mini {
namespace polynomial {

template <int kDegree>
struct Derivative {
 public:
  static double GetValue(double x) {
    constexpr auto a_prev = (2.0 * kDegree - 1) / (kDegree - 1);
    constexpr auto a_prev_prev = kDegree / (kDegree - 1.0);
    return Derivative<kDegree-1>::GetValue(x) * a_prev * x
         - Derivative<kDegree-2>::GetValue(x) * a_prev_prev;
  }
  using Vector = algebra::Column<double, kDegree+1>;
  static Vector GetAllValues(double x) {
    Vector result{0.0};
    Derivative<kDegree>::FillAllValues(x, &result[kDegree]);
    return result;
  }
  static void FillAllValues(double x, double* result) {
    constexpr auto a_prev = (2.0 * kDegree - 1) / (kDegree - 1);
    constexpr auto a_prev_prev = kDegree / (kDegree - 1.0);
    Derivative<kDegree-1>::FillAllValues(x, result-1);
    *result = *(result-1) * a_prev * x 
            - *(result-2) * a_prev_prev;
  }
};

template <>
struct Derivative<0> {
 public:
  static double GetValue(double x) {
    return 0.0;
  }
  using Vector = algebra::Column<double, 1>;
  static Vector GetAllValues(double x) {
    Vector result{0.0};
    return result;
  }
  static void FillAllValues(double x, double* result) {
    *result = 0.0;
  }
};

template <>
struct Derivative<1> {
 public:
  static double GetValue(double x) {
    return 1.0;
  }
  using Vector = algebra::Column<double, 2>;
  static Vector GetAllValues(double x) {
    Vector result{0.0, 1.0};
    return result;
  }
  static void FillAllValues(double x, double* result) {
    *result = 1.0;
  }
};

}  // namespace polynomial
}  // namespace mini

#endif  // MINI_POLYNOMIAL_LEGENDRE_HPP_