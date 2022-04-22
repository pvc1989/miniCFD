// Copyright 2019 Weicheng Pei and Minghao Yang

#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "mini/algebra/column.hpp"
#include "mini/polynomial/legendre.hpp"


namespace mini {
namespace polynomial {

void ExpectNearZero(double x) { EXPECT_NEAR(x, 0.0, 1e-15); }
// Generic comparison:
template <int kDegrees>
bool Compare(double* v, double x) {
  static_assert(kDegrees >= 0);
  ExpectNearZero(*v - Legendre<kDegrees>::GetValue(x));
  return kDegrees > 0 ? Compare<kDegrees-1>(v-1, x) : true;
}
template <>
bool Compare<0>(double* v, double x) {
  ExpectNearZero(*v - 1.0);
  return true;
}
class TestLegendre : public ::testing::Test {
};
TEST_F(TestLegendre, GetValueAtRoots) {
  double e = 1e-15;
  // P_{0}(x) = 1
  ExpectNearZero(Legendre<0>::GetValue(0) - 1);
  // P_{1}(x) = x
  ExpectNearZero(Legendre<1>::GetValue(0));
  // P_{2}(x) = (x^2 - 3) / 2.0
  ExpectNearZero(Legendre<2>::GetValue(+std::sqrt(1.0/3)));
  ExpectNearZero(Legendre<2>::GetValue(-std::sqrt(1.0/3)));
  // P_{3}(x) = 1/2 * (5 * x^3 - 3 * x)
  ExpectNearZero(Legendre<3>::GetValue(0));
  ExpectNearZero(Legendre<3>::GetValue(+std::sqrt(3.0/5)));
  ExpectNearZero(Legendre<3>::GetValue(-std::sqrt(3.0/5)));
  // P_{4}(x) = 1/8 * (35 * x^4 - 30 * x^2 + 3)
  double r_1 = std::sqrt(3.0/7 - 2.0/7*std::sqrt(6.0/5));
  ExpectNearZero(Legendre<4>::GetValue(+r_1));
  ExpectNearZero(Legendre<4>::GetValue(-r_1));
  double r_3 = std::sqrt(3.0/7 + 2.0/7*std::sqrt(6.0/5));
  ExpectNearZero(Legendre<4>::GetValue(+r_3));
  ExpectNearZero(Legendre<4>::GetValue(-r_3));
  // P_{5}(x) = 1/8 * (63 * x^5 - 70 * x^3 + 15 * x)
  ExpectNearZero(Legendre<5>::GetValue(0));
  r_1 = std::sqrt(5 - 2*std::sqrt(10.0/7)) / 3;
  ExpectNearZero(Legendre<5>::GetValue(+r_1));
  ExpectNearZero(Legendre<5>::GetValue(-r_1));
  r_3 = std::sqrt(5 + 2*std::sqrt(10.0/7)) / 3;
  ExpectNearZero(Legendre<5>::GetValue(+r_3));
  ExpectNearZero(Legendre<5>::GetValue(-r_3));
}
TEST_F(TestLegendre, GetAllValues) {
  std::vector<double> x_array;
  double x = -1.0;
  while (x <= 1.0) {
    x_array.emplace_back(x);
    x += 0.1;
  }
  for (auto x : x_array) {
    constexpr int kDegrees = 8;
    auto values = Legendre<kDegrees>::GetAllValues(x);
    Compare<kDegrees>(&values[kDegrees], x);
  }
}

}  // namespace polynomial
}  // namespace mini
