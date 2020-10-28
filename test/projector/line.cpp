// Copyright 2019 Weicheng Pei and Minghao Yang

#include <cmath>

#include "mini/projector/line.hpp"
#include "mini/polynomial/legendre.hpp"

#include "gtest/gtest.h"

namespace mini {
namespace projector {

class TestLine : public ::testing::Test {
 protected:
  using Projector = Line<4>;
  using P2 = polynomial::Legendre<2>;
  static constexpr double epsilon = 1e-15;
};
TEST_F(TestLine, LegendreOnStandardLine) {
  auto projector = Projector({-1}, {+1});
  auto function = [](double x){ return P2::GetValue(x); };
  auto coefficients = projector.GetCoefficients(function);
  EXPECT_NEAR(coefficients[0], 0, epsilon);
  EXPECT_NEAR(coefficients[1], 0, epsilon);
  EXPECT_NEAR(coefficients[2], 1, epsilon);
  EXPECT_NEAR(coefficients[3], 0, epsilon);
}
TEST_F(TestLine, LegendreOnShiftedLine) {
  auto projector = Projector({0}, {+2});
  auto function = [](double x){ return P2::GetValue(x-1); };
  auto coefficients = projector.GetCoefficients(function);
  EXPECT_NEAR(coefficients[0], 0, epsilon);
  EXPECT_NEAR(coefficients[1], 0, epsilon);
  EXPECT_NEAR(coefficients[2], 1, epsilon);
  EXPECT_NEAR(coefficients[3], 0, epsilon);
}
TEST_F(TestLine, LegendreOnScaledLine) {
  auto projector = Projector({-2}, {+2});
  auto function = [](double x){ return P2::GetValue(x/2); };
  auto coefficients = projector.GetCoefficients(function);
  EXPECT_NEAR(coefficients[0], 0, epsilon);
  EXPECT_NEAR(coefficients[1], 0, epsilon);
  EXPECT_NEAR(coefficients[2], 1, epsilon);
  EXPECT_NEAR(coefficients[3], 0, epsilon);
}

}  // namespace projector
}  // namespace mini
