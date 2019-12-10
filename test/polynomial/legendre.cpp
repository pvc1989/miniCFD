// Copyright 2019 Weicheng Pei and Minghao Yang

#include <cmath>
#include <vector>

#include "gtest/gtest.h"

#include "mini/algebra/column.hpp"
#include "mini/polynomial/legendre.hpp"


namespace mini {
namespace polynomial {

class LegendreTest : public ::testing::Test {
 protected: 
  static constexpr int kDegree = 4;
  static constexpr int kDof = kDegree + 1;
  using Scalar = double;
  using Vector = algebra::Column<Scalar, kDof>;
};
TEST_F(LegendreTest, LegendreSingle) {
  double e = 1e-15;
  EXPECT_DOUBLE_EQ(std::abs(legendre<0>(0)), 1);
  EXPECT_LT(std::abs(legendre<1>(0)), e);
  EXPECT_LT(std::abs(legendre<2>(+std::sqrt(1.0/3))), e);
  EXPECT_LT(std::abs(legendre<2>(-std::sqrt(1.0/3))), e);
  EXPECT_LT(std::abs(legendre<3>(0)), e);
  EXPECT_LT(std::abs(legendre<3>(+std::sqrt(3.0/5))), e);
  EXPECT_LT(std::abs(legendre<3>(-std::sqrt(3.0/5))), e);
  EXPECT_LT(std::abs(legendre<4>(+std::sqrt(3.0/7 - 2.0/7*std::sqrt(6.0/5)))), e);
  EXPECT_LT(std::abs(legendre<4>(-std::sqrt(3.0/7 - 2.0/7*std::sqrt(6.0/5)))), e);
  EXPECT_LT(std::abs(legendre<4>(+std::sqrt(3.0/7 + 2.0/7*std::sqrt(6.0/5)))), e);
  EXPECT_LT(std::abs(legendre<4>(-std::sqrt(3.0/7 + 2.0/7*std::sqrt(6.0/5)))), e);
  EXPECT_LT(std::abs(legendre<5>(0)), e);
  EXPECT_LT(std::abs(legendre<5>(+std::sqrt(5 - 2*std::sqrt(10.0/7)) / 3)), e);
  EXPECT_LT(std::abs(legendre<5>(-std::sqrt(5 - 2*std::sqrt(10.0/7)) / 3)), e);
  EXPECT_LT(std::abs(legendre<5>(+std::sqrt(5 + 2*std::sqrt(10.0/7)) / 3)), e);
  EXPECT_LT(std::abs(legendre<5>(-std::sqrt(5 + 2*std::sqrt(10.0/7)) / 3)), e);
}

TEST_F(LegendreTest, LegendreArray) {
  std::vector<double> x_array;
  double x = -1.0;
  while (x <= 1.0) {
    x_array.emplace_back(x);
    x += 0.1;
  }
  for (auto x : x_array) {
    auto value = legendre_array<kDegree>(x);  
    EXPECT_DOUBLE_EQ(value[0], legendre<0>(x));
    EXPECT_DOUBLE_EQ(value[1], legendre<1>(x));
    EXPECT_DOUBLE_EQ(value[2], legendre<2>(x));
    EXPECT_DOUBLE_EQ(value[3], legendre<3>(x));
    EXPECT_DOUBLE_EQ(value[4], legendre<4>(x));
  }
}

}  // namespace basis
}  // namespace mini