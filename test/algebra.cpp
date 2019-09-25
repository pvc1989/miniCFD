// Copyright 2019 Weicheng Pei and Minghao Yang
#include "gtest/gtest.h"

#include "mini/algebra/column.hpp"
#include "mini/algebra/matrix.hpp"

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

class MatrixTest : public ::testing::Test {
 protected:
  using Matrix = Matrix<int, 2, 2>;
  using Column = Column<int, 2>;
};
TEST_F(MatrixTest, TestConstuctors) {
  { Matrix m; }
  { Matrix m{{1, 2}, {3, 4}}; }
  { Matrix m = {{1, 2}, {3, 4}}; }
  { auto m = Matrix(); }
  { auto m = Matrix{}; }
  { auto m = Matrix{{1, 2}, {3, 4}}; }
}
TEST_F(MatrixTest, TestIndexing) {
  auto m = Matrix{{1, 2}, {3, 4}};
  EXPECT_EQ(m[0][0], 1);
  EXPECT_EQ(m[0][1], 2);
  EXPECT_EQ(m[1][0], 3);
  EXPECT_EQ(m[1][1], 4);
}
TEST_F(MatrixTest, TestMatrixProduct) {
  auto m = Matrix{{1, 2}, {3, 4}};
  auto v = Column{5, 6};
  auto p = m * v;
  EXPECT_EQ(p[0], m[0][0]*v[0] + m[0][1]*v[1]);
  EXPECT_EQ(p[1], m[1][0]*v[0] + m[1][1]*v[1]);
}

}  // namespace algebra
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
