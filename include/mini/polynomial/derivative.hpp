//  Copyright 2019 PEI Weicheng and YANG Minghao

#ifndef MINI_POLYNOMIAL_DERIVATIVE_HPP_
#define MINI_POLYNOMIAL_DERIVATIVE_HPP_

#include "mini/algebra/column.hpp"

namespace mini {
namespace polynomial {

template <int kDegrees>
struct Derivative {
 public:
  static double GetValue(double x) {
    constexpr auto a_prev = (2.0 * kDegrees - 1) / (kDegrees - 1);
    constexpr auto a_prev_prev = kDegrees / (kDegrees - 1.0);
    return Derivative<kDegrees-1>::GetValue(x) * a_prev * x
         - Derivative<kDegrees-2>::GetValue(x) * a_prev_prev;
  }
  using Vector = algebra::Column<double, kDegrees+1>;
  static Vector GetAllValues(double x) {
    Vector result{0.0};
    Derivative<kDegrees>::FillAllValues(x, &result[kDegrees]);
    return result;
  }
  static void FillAllValues(double x, double* result) {
    constexpr auto a_prev = (2.0 * kDegrees - 1) / (kDegrees - 1);
    constexpr auto a_prev_prev = kDegrees / (kDegrees - 1.0);
    Derivative<kDegrees-1>::FillAllValues(x, result-1);
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

#endif  // MINI_POLYNOMIAL_DERIVATIVE_HPP_
