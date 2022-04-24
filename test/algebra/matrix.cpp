// Copyright 2019 PEI Weicheng and YANG Minghao

#include "mini/algebra/matrix.hpp"
#include "mini/algebra/column.hpp"
#include "gtest/gtest.h"

namespace mini {
namespace algebra {

class TestMatrix : public ::testing::Test {
 protected:
  using M = Matrix<int, 2, 2>;
  using C = Column<int, 2>;
};
TEST_F(TestMatrix, Constuctors) {
  { M m; }
  { M m{{1, 2}, {3, 4}}; }
  { M m = {{1, 2}, {3, 4}}; }
  { auto m = M(); }
  { auto m = M{}; }
  { auto m = M{{1, 2}, {3, 4}}; }
}
TEST_F(TestMatrix, Indexing) {
  auto m = M{{1, 2}, {3, 4}};
  EXPECT_EQ(m[0][0], 1);
  EXPECT_EQ(m[0][1], 2);
  EXPECT_EQ(m[1][0], 3);
  EXPECT_EQ(m[1][1], 4);
}
TEST_F(TestMatrix, Product) {
  auto m = M{{1, 2}, {3, 4}};
  auto c = C{5, 6};
  auto p = m * c;
  EXPECT_EQ(p[0], m[0][0]*c[0] + m[0][1]*c[1]);
  EXPECT_EQ(p[1], m[1][0]*c[0] + m[1][1]*c[1]);
}
TEST_F(TestMatrix, MultiplyingWithScalar) {
  auto m = M{{1, 2}, {3, 4}};
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
