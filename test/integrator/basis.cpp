//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include "mini/integrator/basis.hpp"

#include "gtest/gtest.h"

class TestRawBasis : public ::testing::Test {
};
TEST_F(TestRawBasis, In2dSpace) {
  using Basis = mini::integrator::RawBasis<double, 2, 2>;
  static_assert(Basis::N == 6);
  double x, y;
  typename Basis::MatNx1 res;
  res = Basis::CallAt({x, y});
  EXPECT_EQ(res[0], 1);
  EXPECT_EQ(res[1], x);
  EXPECT_EQ(res[2], y);
  EXPECT_EQ(res[3], x * x);
  EXPECT_EQ(res[4], x * y);
  EXPECT_EQ(res[5], y * y);
  x = 0.3; y = 0.4;
  res = Basis::CallAt({x, y});
  EXPECT_EQ(res[0], 1);
  EXPECT_EQ(res[1], x);
  EXPECT_EQ(res[2], y);
  EXPECT_EQ(res[3], x * x);
  EXPECT_EQ(res[4], x * y);
  EXPECT_EQ(res[5], y * y);
}
TEST_F(TestRawBasis, In3dSpace) {
  using Basis = mini::integrator::RawBasis<double, 3, 2>;
  static_assert(Basis::N == 10);
  double x, y, z;
  typename Basis::MatNx1 res;
  res = Basis::CallAt({x, y, z});
  EXPECT_EQ(res[0], 1);
  EXPECT_EQ(res[1], x);
  EXPECT_EQ(res[2], y);
  EXPECT_EQ(res[3], z);
  EXPECT_EQ(res[4], x * x);
  EXPECT_EQ(res[5], x * y);
  EXPECT_EQ(res[6], x * z);
  EXPECT_EQ(res[7], y * y);
  EXPECT_EQ(res[8], y * z);
  EXPECT_EQ(res[9], z * z);
  x = 0.3; y = 0.4, z = 0.5;
  res = Basis::CallAt({x, y, z});
  EXPECT_EQ(res[0], 1);
  EXPECT_EQ(res[1], x);
  EXPECT_EQ(res[2], y);
  EXPECT_EQ(res[3], z);
  EXPECT_EQ(res[4], x * x);
  EXPECT_EQ(res[5], x * y);
  EXPECT_EQ(res[6], x * z);
  EXPECT_EQ(res[7], y * y);
  EXPECT_EQ(res[8], y * z);
  EXPECT_EQ(res[9], z * z);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
