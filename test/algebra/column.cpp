// Copyright 2019 Weicheng Pei and Minghao Yang
#include "mini/algebra/column.hpp"
#include "gtest/gtest.h"

namespace mini {
namespace algebra {

class ColumnTest : public ::testing::Test {
 protected:
  using Vector = Column<int, 3>;
};
TEST_F(ColumnTest, TestConstuctors) {
  { Vector v; }
  { Vector v{1, 2, 3}; }
  { Vector v = {1, 2, 3}; }
  { auto v = Vector(); }
  { auto v = Vector{}; }
  { auto v = Vector{1, 2, 3}; }
}
TEST_F(ColumnTest, TestOperatorEqual) {
  auto u = Vector{1, 2, 3};
  auto v = Vector{1, 2, 3};
  EXPECT_EQ(u, v);
  auto w = Vector{0, 0, 0};
  EXPECT_NE(u, w);
}
TEST_F(ColumnTest, TestPlusAndMinus) {
  auto u = Vector{1, 2, 3};
  auto v = Vector{0, 0, 0};
  EXPECT_EQ(u + v, u);
  EXPECT_EQ(v + v, v);
  EXPECT_EQ(u - v, u);
  EXPECT_EQ(u - u, v);
}
TEST_F(ColumnTest, TestScalarMultiplication) {
  auto u = Vector{1, 2, 3};
  EXPECT_EQ(u * 1, u);
  EXPECT_EQ(1 * u, u);
  auto v = Vector{0, 0, 0};
  EXPECT_EQ(u * 0, v);
  EXPECT_EQ(0 * u, v);
}
TEST_F(ColumnTest, TestDotProduct) {
  auto u = Vector{0, 1, 2};
  auto v = Vector{0, 0, 0};
  EXPECT_EQ(u.Dot(u), 5);
  EXPECT_EQ(u.Dot(v), 0);
  EXPECT_EQ(v.Dot(v), 0);
  EXPECT_EQ(v.Dot(u), 0);
}

}  // namespace algebra
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
