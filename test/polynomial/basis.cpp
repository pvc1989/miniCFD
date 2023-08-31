//  Copyright 2021 PEI Weicheng and JIANG Yuyan

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
#include "mini/polynomial/basis.hpp"

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

class TestLinearBasis : public ::testing::Test {
};
TEST_F(TestLinearBasis, In2dSpace) {
  using Basis = mini::polynomial::Linear<double, 2, 2>;
  auto basis = Basis({0, 0});
  static_assert(Basis::N == 6);
  std::srand(31415926);
  double x{rand_f()}, y{rand_f()};
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
TEST_F(TestLinearBasis, In3dSpace) {
  using Basis = mini::polynomial::Linear<double, 3, 2>;
  auto basis = Basis({0, 0, 0});
  static_assert(Basis::N == 10);
  std::srand(31415926);
  double x{rand_f()}, y{rand_f()}, z{rand_f()};
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

class TestOrthoNormalBasis : public ::testing::Test {
};
TEST_F(TestOrthoNormalBasis, OnTriangle) {
  using Lagrange = mini::lagrange::Triangle3<double, 2>;
  using Gauss = mini::gauss::Triangle<double, 2, 16>;
  using Coord = Gauss::Global;
  Coord p0{0, 0}, p1{3, 0}, p2{0, 3};
  auto lagrange = Lagrange{ p0, p1, p2 };
  auto gauss = Gauss(lagrange);
  using Basis = mini::polynomial::OrthoNormal<double, 2, 2>;
  auto basis = Basis(gauss);
  EXPECT_DOUBLE_EQ(gauss.area(), basis.Measure());
  std::cout << basis.coeff() << std::endl;
  auto area = mini::gauss::Integrate(
        [](const Coord &){ return 1.0; }, basis.GetGauss());
  EXPECT_DOUBLE_EQ(basis.Measure(), area);
  auto f = [&basis](const Coord &coord){
      return Basis::MatNxN(basis(coord) * basis(coord).transpose());
  };
  Basis::MatNxN diff = mini::gauss::Integrate(f, basis.GetGauss())
      - Basis::MatNxN::Identity();
  EXPECT_NEAR(diff.norm(), 0.0, 1e-13);
}
TEST_F(TestOrthoNormalBasis, OnQuadrangle) {
  using Lagrange = mini::lagrange::Quadrangle4<double, 2>;
  using Gauss = mini::gauss::Quadrangle<double, 2, 4, 4>;
  using Coord = Gauss::Global;
  Coord p0{-1, -1}, p1{+1, -1}, p2{+1, +1}, p3{-1, +1};
  auto lagrange = Lagrange(p0, p1, p2, p3);
  auto gauss = Gauss(lagrange);
  using Basis = mini::polynomial::OrthoNormal<double, 2, 2>;
  auto basis = Basis(gauss);
  EXPECT_DOUBLE_EQ(gauss.area(), basis.Measure());
  std::cout << basis.coeff() << std::endl;
  auto area = mini::gauss::Integrate(
        [](const Coord &){ return 1.0; }, basis.GetGauss());
  EXPECT_DOUBLE_EQ(basis.Measure(), area);
  auto f = [&basis](const Coord &coord){
      return Basis::MatNxN(basis(coord) * basis(coord).transpose());
  };
  Basis::MatNxN diff = mini::gauss::Integrate(f, basis.GetGauss())
      - Basis::MatNxN::Identity();
  EXPECT_NEAR(diff.norm(), 0.0, 1e-14);
}
TEST_F(TestOrthoNormalBasis, OnTetrahedron) {
  using Gauss = mini::gauss::Tetrahedron<double, 24>;
  using Lagrange = mini::lagrange::Tetrahedron4<double>;
  using Coord = Gauss::Global;
  Coord p0{0, 0, 0}, p1{3, 0, 0}, p2{0, 3, 0}, p3{0, 0, 3};
  auto lagrange = Lagrange(p0, p1, p2, p3);
  auto gauss = Gauss(lagrange);
  using Basis = mini::polynomial::OrthoNormal<double, 3, 2>;
  auto basis = Basis(gauss);
  EXPECT_DOUBLE_EQ(gauss.volume(), basis.Measure());
  std::cout << basis.coeff() << std::endl;
  auto volume = mini::gauss::Integrate(
        [](const Coord &){ return 1.0; }, basis.GetGauss());
  EXPECT_DOUBLE_EQ(basis.Measure(), volume);
  auto f = [&basis](const Coord &coord){
      return Basis::MatNxN(basis(coord) * basis(coord).transpose());
  };
  Basis::MatNxN diff = mini::gauss::Integrate(f, basis.GetGauss())
      - Basis::MatNxN::Identity();
  EXPECT_NEAR(diff.norm(), 0.0, 1e-14);
}
TEST_F(TestOrthoNormalBasis, OnHexahedron) {
  using Lagrange = mini::lagrange::Hexahedron8<double>;
  using Gauss = mini::gauss::Hexahedron<double, 4, 4, 4>;
  using Coord = Gauss::Global;
  Coord p0{-1, -1, -1}, p1{+1, -1, -1}, p2{+1, +1, -1}, p3{-1, +1, -1},
        p4{-1, -1, +1}, p5{+1, -1, +1}, p6{+1, +1, +1}, p7{-1, +1, +1};
  auto lagrange = Lagrange{ p0, p1, p2, p3, p4, p5, p6, p7 };
  auto gauss = Gauss(lagrange);
  using Basis = mini::polynomial::OrthoNormal<double, 3, 2>;
  auto basis = Basis(gauss);
  EXPECT_DOUBLE_EQ(gauss.volume(), basis.Measure());
  std::cout << basis.coeff() << std::endl;
  auto volume = mini::gauss::Integrate(
        [](const Coord &){ return 1.0; }, basis.GetGauss());
  EXPECT_DOUBLE_EQ(basis.Measure(), volume);
  auto f = [&basis](const Coord &coord){
      return Basis::MatNxN(basis(coord) * basis(coord).transpose());
  };
  Basis::MatNxN diff = mini::gauss::Integrate(f, basis.GetGauss())
      - Basis::MatNxN::Identity();
  EXPECT_NEAR(diff.norm(), 0.0, 1e-14);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
