// Copyright 2019 Weicheng Pei and Minghao Yang
#include "mini/element/point.hpp"

#include <vector>

#include "gtest/gtest.h"

namespace mini {
namespace element {

class PointTest : public ::testing::Test {
 protected:
  using Real = double;
  using P1 = Point<Real, 1>;
  using P2 = Point<Real, 2>;
  using P3 = Point<Real, 3>;
  const P1::IndexType i{8};
  const Real x{1.0}, y{2.0}, z{3.0};
};
TEST_F(PointTest, TemplateConstructor) {
  // Test P1(Index i, Real x):
  auto p1 = P1(i, x);
  EXPECT_EQ(p1.I(), i);
  EXPECT_EQ(p1.X(), x);
  // Test P2(Index i, Real x, Real y):
  auto p2 = P2(i, x, y);
  EXPECT_EQ(p2.I(), i);
  EXPECT_EQ(p2.X(), x);
  EXPECT_EQ(p2.Y(), y);
  // Test P3(Index i, Real x, Real y, Real z):
  auto p3 = P3(i, x, y, z);
  EXPECT_EQ(p3.I(), i);
  EXPECT_EQ(p3.X(), x);
  EXPECT_EQ(p3.Y(), y);
  EXPECT_EQ(p3.Z(), z);
}
TEST_F(PointTest, InitializerListConstructor) {
  // Test P1(Index, std::initializer_list<Real>):
  auto p1 = P1(i, {x});
  EXPECT_EQ(p1.I(), i);
  EXPECT_EQ(p1.X(), x);
  // Test P1(Index, std::initializer_list<Real>):
  auto p2 = P2(i, {x, y});
  EXPECT_EQ(p2.I(), i);
  EXPECT_EQ(p2.X(), x);
  EXPECT_EQ(p2.Y(), y);
  // Test P1(Index, std::initializer_list<Real>):
  auto p3 = P3(i, {x, y, z});
  EXPECT_EQ(p3.I(), i);
  EXPECT_EQ(p3.X(), x);
  EXPECT_EQ(p3.Y(), y);
  EXPECT_EQ(p3.Z(), z);
}

}  // namespace element
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
