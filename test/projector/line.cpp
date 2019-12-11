// Copyright 2019 Weicheng Pei and Minghao Yang
#include "mini/geometry/dim0.hpp"
#include "mini/projector/line.hpp"

#include <cmath>

#include "gtest/gtest.h"


namespace mini {
namespace projector {

class Line1dTest : public ::testing::Test {
 protected:
  static constexpr int kDegree = 4;
  using Point = geometry::Point<double, 1>;
};
TEST_F(Line1dTest, SecondOrder) {
  Point head{-1}, tail{+1};
  auto line = Line<3>(head, tail);
  auto function = [](double x){ return 0.5 * (3*x*x - 1); };
  auto coefficients = line.GetCoefficients(function);
  EXPECT_DOUBLE_EQ(coefficients[0], 0);
  EXPECT_DOUBLE_EQ(coefficients[1], 0);
  EXPECT_DOUBLE_EQ(coefficients[2], 1);
  EXPECT_DOUBLE_EQ(coefficients[3], 0);
}

}  // namespace projector
}  // namespace mini
