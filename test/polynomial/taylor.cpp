//  Copyright 2023 PEI Weicheng

#include <iostream>
#include <cstdlib>

#include "mini/gauss/function.hpp"
#include "mini/gauss/tetrahedron.hpp"
#include "mini/lagrange/tetrahedron.hpp"
#include "mini/gauss/hexahedron.hpp"
#include "mini/lagrange/hexahedron.hpp"
#include "mini/gauss/triangle.hpp"
#include "mini/lagrange/triangle.hpp"
#include "mini/gauss/quadrangle.hpp"
#include "mini/lagrange/quadrangle.hpp"
#include "mini/polynomial/taylor.hpp"

#include "gtest/gtest.h"

double rand_f() {
  return std::rand() / (1.0 + RAND_MAX);
}

class TestTaylorBasis : public ::testing::Test {
};
TEST_F(TestTaylorBasis, In2dSpace) {
  using Basis = mini::polynomial::Taylor<double, 2, 2>;
  static_assert(Basis::N == 6);
  std::srand(31415926);
  double x{rand_f()}, y{rand_f()};
  typename Basis::MatNx1 res;
  res = Basis::GetValue({x, y});
  EXPECT_EQ(res[0], 1);
  EXPECT_EQ(res[1], x);
  EXPECT_EQ(res[2], y);
  EXPECT_EQ(res[3], x * x);
  EXPECT_EQ(res[4], x * y);
  EXPECT_EQ(res[5], y * y);
  x = 0.3; y = 0.4;
  res = Basis::GetValue({x, y});
  EXPECT_EQ(res[0], 1);
  EXPECT_EQ(res[1], x);
  EXPECT_EQ(res[2], y);
  EXPECT_EQ(res[3], x * x);
  EXPECT_EQ(res[4], x * y);
  EXPECT_EQ(res[5], y * y);
}
TEST_F(TestTaylorBasis, In3dSpace) {
  using Basis = mini::polynomial::Taylor<double, 3, 2>;
  static_assert(Basis::N == 10);
  std::srand(31415926);
  double x{rand_f()}, y{rand_f()}, z{rand_f()};
  typename Basis::MatNx1 res;
  res = Basis::GetValue({x, y, z});
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
  res = Basis::GetValue({x, y, z});
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
