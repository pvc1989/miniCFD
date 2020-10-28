// Copyright 2019 Weicheng Pei and Minghao Yang

#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "mini/algebra/column.hpp"
#include "mini/polynomial/derivative.hpp"

namespace mini {
namespace polynomial {

class DerivativeTest : public ::testing::Test {
 protected:
  void ExpectNearZero(double x) { EXPECT_NEAR(x, 0.0, 1e-12); }
  // Generic comparison:
  template <int kDegree>
  constexpr bool Compare(double* v, double x) {
    static_assert(kDegree >= 0);
    ExpectNearZero(*v - Derivative<kDegree>::GetValue(x));
    return kDegree > 0 ? Compare<kDegree-1>(v-1, x) : true;
  }
  template <>
  constexpr bool Compare<0>(double* v, double x) {
    ExpectNearZero(*v - 0.0);
    return true;
  }
  double Dp0(double x) {
    return 0.0;
  }
  double Dp1(double x) {
    return 1.0;
  }
  double Dp2(double x) {
    return 3.0 * x;
  }
  double Dp3(double x) {
    return (15 * x*x - 3) / 2;
  }
  double Dp4(double x) {
    return (35 * std::pow(x, 3) - 15 * x) / 2;
  }
  double Dp5(double x) {
    return (315 * std::pow(x, 4) - 210 * x*x + 15) / 8;
  }
};
TEST_F(DerivativeTest, GetValue) {
  std::vector<double> x_array;
  double x = -1.0;
  while (x <= 1.0) {
    x_array.emplace_back(x);
    x += 0.1;
  }
  // P'_{0}(x) = 0
  for (auto x : x_array) {
    EXPECT_DOUBLE_EQ(Derivative<0>::GetValue(x), Dp0(x));
  }
  // P'_{1}(x) = 1
  for (auto x : x_array) {
    EXPECT_DOUBLE_EQ(Derivative<1>::GetValue(x), Dp1(x));
  }
  // P'_{2}(x) = 3 * x
  for (auto x : x_array) {
    EXPECT_DOUBLE_EQ(Derivative<2>::GetValue(x), Dp2(x));
  }
  // P'_{3}(x) = (15 * x^2 - 3) / 2
  for (auto x : x_array) {
    EXPECT_DOUBLE_EQ(Derivative<3>::GetValue(x), Dp3(x));
  }
  // P'_{4}(x) = (35 * x^3 - 15 * x) / 2
  for (auto x : x_array) {
    EXPECT_DOUBLE_EQ(Derivative<4>::GetValue(x), Dp4(x));
  }
  // P'_{5}(x) = (315 * x^4 - 210 * x^2 + 15) / 8
  for (auto x : x_array) {
    EXPECT_DOUBLE_EQ(Derivative<5>::GetValue(x), Dp5(x));
  }
}
TEST_F(DerivativeTest, GetAllValues) {
  std::vector<double> x_array;
  double x = -1.0;
  while (x <= 1.0) {
    x_array.emplace_back(x);
    x += 0.1;
  }
  for (auto x : x_array) {
    constexpr int kDegree = 8;
    auto values = Derivative<kDegree>::GetAllValues(x);
    Compare<kDegree>(&values[kDegree], x);
  }
}

}  // namespace polynomial
}  // namespace mini
