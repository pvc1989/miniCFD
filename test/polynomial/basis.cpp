//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <iostream>

#include "mini/integrator/function.hpp"
#include "mini/integrator/hexa.hpp"
#include "mini/integrator/quad.hpp"
#include "mini/polynomial/basis.hpp"

#include "gtest/gtest.h"

class TestRaw : public ::testing::Test {
};
TEST_F(TestRaw, In2dSpace) {
  using Basis = mini::polynomial::Raw<double, 2, 2>;
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
TEST_F(TestRaw, In3dSpace) {
  using Basis = mini::polynomial::Raw<double, 3, 2>;
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

class TestBasis : public ::testing::Test {
};
TEST_F(TestBasis, In2dSpace) {
  using Basis = mini::polynomial::Linear<double, 2, 2>;
  auto basis = Basis({0, 0});
  static_assert(Basis::N == 6);
  double x, y;
  typename Basis::MatNx1 res;
  res = basis({x, y});
  EXPECT_EQ(res[0], 1);
  EXPECT_EQ(res[1], x);
  EXPECT_EQ(res[2], y);
  EXPECT_EQ(res[3], x * x);
  EXPECT_EQ(res[4], x * y);
  EXPECT_EQ(res[5], y * y);
  x = 0.3; y = 0.4;
  res = basis({x, y});
  EXPECT_EQ(res[0], 1);
  EXPECT_EQ(res[1], x);
  EXPECT_EQ(res[2], y);
  EXPECT_EQ(res[3], x * x);
  EXPECT_EQ(res[4], x * y);
  EXPECT_EQ(res[5], y * y);
}
TEST_F(TestBasis, In3dSpace) {
  using Basis = mini::polynomial::Linear<double, 3, 2>;
  auto basis = Basis({0, 0, 0});
  static_assert(Basis::N == 10);
  double x, y, z;
  typename Basis::MatNx1 res;
  res = basis({x, y, z});
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
  res = basis({x, y, z});
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

class TestOrthoNormal : public ::testing::Test {
};
TEST_F(TestOrthoNormal, In2dSpace) {
  using Gauss = mini::integrator::Quad<double, 2, 4, 4>;
  using Coord = Gauss::GlobalCoord;
  Coord p0{-1, -1}, p1{+1, -1}, p2{+1, +1}, p3{-1, +1};
  auto gauss = Gauss(p0, p1, p2, p3);
  using Basis = mini::polynomial::OrthoNormal<double, 2, 2>;
  auto basis = Basis(gauss);
  std::cout << basis.coeff() << std::endl;
  auto area = mini::integrator::Integrate(
        [](const Coord &){ return 1.0; }, basis.GetGauss());
  EXPECT_DOUBLE_EQ(basis.Measure(), area);
}
TEST_F(TestOrthoNormal, In3dSpace) {
  using Gauss = mini::integrator::Hexa<double, 4, 4, 4>;
  using Coord = Gauss::GlobalCoord;
  Coord p0{-1, -1, -1}, p1{+1, -1, -1}, p2{+1, +1, -1}, p3{-1, +1, -1},
        p4{-1, -1, +1}, p5{+1, -1, +1}, p6{+1, +1, +1}, p7{-1, +1, +1};
  auto gauss = Gauss(p0, p1, p2, p3, p4, p5, p6, p7);
  using Basis = mini::polynomial::OrthoNormal<double, 3, 2>;
  auto basis = Basis(gauss);
  std::cout << basis.coeff() << std::endl;
  auto volume = mini::integrator::Integrate(
        [](const Coord &){ return 1.0; }, basis.GetGauss());
  EXPECT_DOUBLE_EQ(basis.Measure(), volume);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
