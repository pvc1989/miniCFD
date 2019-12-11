// Copyright 2019 Weicheng Pei and Minghao Yang
#include "mini/algebra/matrix.hpp"
#include "mini/algebra/column.hpp"
#include "gtest/gtest.h"

namespace mini {
namespace algebra {

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
TEST_F(MatrixTest, TestScalarMultiplication) {
  auto m = Matrix{{1, 2}, {3, 4}};
  auto p = m * 2;
  EXPECT_EQ(p[0][0], 2);
  EXPECT_EQ(p[0][1], 4);
  EXPECT_EQ(p[1][0], 6);
  EXPECT_EQ(p[1][1], 8);
}

}  // namespace algebra
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
