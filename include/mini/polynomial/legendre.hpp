//  Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_POLYNOMIAL_LEGENDRE_HPP_
#define MINI_POLYNOMIAL_LEGENDRE_HPP_

#include "mini/algebra/column.hpp"

namespace mini {
namespace polynomial {

template <int kDegree>
struct Legendre {
 public:
  static double GetValue(double x) {
    constexpr auto a_prev = (2.0 * kDegree - 1) / kDegree;
    constexpr auto a_prev_prev = (1.0 - kDegree) / kDegree;
    return Legendre<kDegree-1>::GetValue(x) * a_prev * x
         + Legendre<kDegree-2>::GetValue(x) * a_prev_prev;
  }
  using Vector = algebra::Column<double, kDegree+1>;
  static Vector GetAllValues(double x) {
    Vector result{1.0};
    Legendre<kDegree>::FillAllValues(x, &result[kDegree]);
    return result;
  }
  static void FillAllValues(double x, double* result) {
    constexpr auto a_prev = (2.0 * kDegree - 1) / kDegree;
    constexpr auto a_prev_prev = (1.0 - kDegree) / kDegree;
    Legendre<kDegree-1>::FillAllValues(x, result-1);
    *result = *(result-1) * a_prev * x 
            + *(result-2) * a_prev_prev;
  }
};

template <>
struct Legendre<0> {
 public:
  static double GetValue(double x) {
    return 1.0;
  }
  using Vector = algebra::Column<double, 1>;
  static Vector GetAllValues(double x) {
    Vector result{1.0};
    return result;
  }
  static void FillAllValues(double x, double* result) {
    *result = 1.0;
  }
};

template <>
struct Legendre<1> {
 public:
  static double GetValue(double x) {
    return x;
  }
  using Vector = algebra::Column<double, 2>;
  static Vector GetAllValues(double x) {
    Vector result{1.0, x};
    return result;
  }
  static void FillAllValues(double x, double* result) {
    *result = x;
  }
};

static constexpr std::array<double, 6> norms{(2 * 0 + 1.0) / 2,
                                             (2 * 1 + 1.0) / 2,
                                             (2 * 2 + 1.0) / 2,
                                             (2 * 3 + 1.0) / 2,
                                             (2 * 4 + 1.0) / 2,
                                             (2 * 5 + 1.0) / 2};

}  // namespace polynomial
}  // namespace mini

#endif  // MINI_POLYNOMIAL_LEGENDRE_HPP_
