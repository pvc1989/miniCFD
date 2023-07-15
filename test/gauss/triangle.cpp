//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cmath>

#include "mini/gauss/function.hpp"
#include "mini/gauss/triangle.hpp"
#include "mini/lagrange/triangle.hpp"

#include "gtest/gtest.h"

class TestGaussTriangle : public ::testing::Test {
};
TEST_F(TestGaussTriangle, OnScaledElementInTwoDimensionalSpace) {
  using Gauss = mini::gauss::Triangle<double, 2, 16>;
  using Lagrange = mini::lagrange::Triangle3<double, 2>;
  using Coord = typename Lagrange::GlobalCoord;
  auto lagrange = Lagrange(
    Coord(0, 0), Coord(2, 0), Coord(2, 2)
  );
  auto gauss = Gauss(lagrange);
  EXPECT_EQ(gauss.CountQuadraturePoints(), 16);
  static_assert(gauss.CellDim() == 2);
  static_assert(gauss.PhysDim() == 2);
  EXPECT_DOUBLE_EQ(gauss.area(), 2.0);
  EXPECT_EQ(gauss.center(), Mat2x1(4./3, 2./3));
  EXPECT_EQ(gauss.LocalToGlobal(1, 0), Coord(0, 0));
  EXPECT_EQ(gauss.LocalToGlobal(0, 1), Coord(2, 0));
  EXPECT_EQ(gauss.LocalToGlobal(0, 0), Coord(2, 2));
  EXPECT_DOUBLE_EQ(Quadrature([](Coord const&){ return 2.0; }, gauss), 1.0);
  EXPECT_DOUBLE_EQ(Integrate([](Coord const&){ return 2.0; }, gauss), 4.0);
  auto f = [](Coord const& xy){ return xy[0]; };
  auto g = [](Coord const& xy){ return xy[1]; };
  auto h = [](Coord const& xy){ return xy[0] * xy[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, gauss), Integrate(h, gauss));
  EXPECT_DOUBLE_EQ(Norm(f, gauss), std::sqrt(Innerprod(f, f, gauss)));
  EXPECT_DOUBLE_EQ(Norm(g, gauss), std::sqrt(Innerprod(g, g, gauss)));
}
TEST_F(TestGaussTriangle, OnMappedElementInThreeDimensionalSpace) {
  using Gauss = mini::gauss::Triangle<double, 3, 16>;
  using Lagrange = mini::lagrange::Triangle3<double, 3>;
  using Local = typename Lagrange::LocalCoord;
  using Global = typename Lagrange::GlobalCoord;
  auto lagrange = Lagrange(
    Global(0, 0, 2), Global(2, 0, 2), Global(2, 2, 2)
  );
  auto gauss = Gauss(lagrange);
  static_assert(gauss.CellDim() == 2);
  static_assert(gauss.PhysDim() == 3);
  EXPECT_DOUBLE_EQ(gauss.area(), 2.0);
  EXPECT_EQ(gauss.center(), Global(4./3, 2./3, 2.));
  EXPECT_EQ(gauss.LocalToGlobal(1, 0), Global(0, 0, 2));
  EXPECT_EQ(gauss.LocalToGlobal(0, 1), Global(2, 0, 2));
  EXPECT_EQ(gauss.LocalToGlobal(0, 0), Global(2, 2, 2));
  EXPECT_DOUBLE_EQ(
      Quadrature([](Mat2x1 const&){ return 2.0; }, gauss), 1.0);
  EXPECT_DOUBLE_EQ(
      Integrate([](Global const&){ return 2.0; }, gauss), 4.0);
  auto f = [](Global const& xyz){ return xyz[0]; };
  auto g = [](Global const& xyz){ return xyz[1]; };
  auto h = [](Global const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, gauss), Integrate(h, gauss));
  EXPECT_DOUBLE_EQ(Norm(f, gauss), std::sqrt(Innerprod(f, f, gauss)));
  EXPECT_DOUBLE_EQ(Norm(g, gauss), std::sqrt(Innerprod(g, g, gauss)));
  // test normal frames
  gauss.BuildNormalFrames();
  for (int q = 0; q < gauss.CountQuadraturePoints(); ++q) {
    auto &frame = gauss.GetNormalFrame(q);
    auto &nu = frame.col(0), &sigma = frame.col(1), &pi = frame.col(2);
    EXPECT_EQ(nu, sigma.cross(pi));
    EXPECT_EQ(sigma, pi.cross(nu));
    EXPECT_EQ(pi, nu.cross(sigma));
    EXPECT_EQ(nu, Global(0, 0, 1));
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
