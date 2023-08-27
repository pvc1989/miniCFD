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
  using Coord = typename Lagrange::Global;
  auto lagrange = Lagrange(
    Coord(0, 0), Coord(2, 0), Coord(2, 2)
  );
  auto gauss = Gauss(lagrange);
  EXPECT_EQ(gauss.CountPoints(), 16);
  static_assert(gauss.CellDim() == 2);
  static_assert(gauss.PhysDim() == 2);
  EXPECT_DOUBLE_EQ(gauss.area(), 2.0);
  EXPECT_NEAR((gauss.center() - Coord(4./3, 2./3)).norm(), 0, 1e-15);
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
  using Local = typename Lagrange::Local;
  using Global = typename Lagrange::Global;
  auto lagrange = Lagrange(
    Global(0, 0, 2), Global(2, 0, 2), Global(2, 2, 2)
  );
  auto gauss = Gauss(lagrange);
  static_assert(gauss.CellDim() == 2);
  static_assert(gauss.PhysDim() == 3);
  EXPECT_DOUBLE_EQ(gauss.area(), 2.0);
  EXPECT_NEAR((gauss.center() - Global(4./3, 2./3, 2.)).norm(), 0, 1e-15);
  EXPECT_EQ(gauss.LocalToGlobal(1, 0), Global(0, 0, 2));
  EXPECT_EQ(gauss.LocalToGlobal(0, 1), Global(2, 0, 2));
  EXPECT_EQ(gauss.LocalToGlobal(0, 0), Global(2, 2, 2));
  EXPECT_DOUBLE_EQ(
      Quadrature([](Local const&){ return 2.0; }, gauss), 1.0);
  EXPECT_DOUBLE_EQ(
      Integrate([](Global const&){ return 2.0; }, gauss), 4.0);
  auto f = [](Global const& xyz){ return xyz[0]; };
  auto g = [](Global const& xyz){ return xyz[1]; };
  auto h = [](Global const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, gauss), Integrate(h, gauss));
  EXPECT_DOUBLE_EQ(Norm(f, gauss), std::sqrt(Innerprod(f, f, gauss)));
  EXPECT_DOUBLE_EQ(Norm(g, gauss), std::sqrt(Innerprod(g, g, gauss)));
  // test normal frames
  Global normal = Global(0, 0, 1).normalized();
  for (int q = 0; q < gauss.CountPoints(); ++q) {
    auto &frame = gauss.GetNormalFrame(q);
    auto &nu = frame[0], &sigma = frame[1], &pi = frame[2];
    EXPECT_NEAR((nu - normal).norm(), 0.0, 1e-15);
    EXPECT_NEAR((nu - sigma.cross(pi)).norm(), 0.0, 1e-15);
    EXPECT_NEAR((sigma - pi.cross(nu)).norm(), 0.0, 1e-15);
    EXPECT_NEAR((pi - nu.cross(sigma)).norm(), 0.0, 1e-15);
  }
}
TEST_F(TestGaussTriangle, OnQuadraticElementInThreeDimensionalSpace) {
  using Gauss = mini::gauss::Triangle<double, 3, 16>;
  using Lagrange = mini::lagrange::Triangle6<double, 3>;
  using Local = typename Lagrange::Local;
  using Global = typename Lagrange::Global;
  auto lagrange = Lagrange(
    Global(0, 0, 2), Global(2, 0, 2), Global(2, 2, 2),
    Global(1, 0, 2), Global(2, 1, 2), Global(1, 1, 2)
  );
  auto gauss = Gauss(lagrange);
  static_assert(gauss.CellDim() == 2);
  static_assert(gauss.PhysDim() == 3);
  EXPECT_DOUBLE_EQ(gauss.area(), 2.0);
  EXPECT_NEAR((gauss.center() - Global(4./3, 2./3, 2.)).norm(), 0, 1e-15);
  EXPECT_EQ(gauss.LocalToGlobal(1, 0), Global(0, 0, 2));
  EXPECT_EQ(gauss.LocalToGlobal(0, 1), Global(2, 0, 2));
  EXPECT_EQ(gauss.LocalToGlobal(0, 0), Global(2, 2, 2));
  EXPECT_DOUBLE_EQ(
      Quadrature([](Local const&){ return 2.0; }, gauss), 1.0);
  EXPECT_DOUBLE_EQ(
      Integrate([](Global const&){ return 2.0; }, gauss), 4.0);
  auto f = [](Global const& xyz){ return xyz[0]; };
  auto g = [](Global const& xyz){ return xyz[1]; };
  auto h = [](Global const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, gauss), Integrate(h, gauss));
  EXPECT_DOUBLE_EQ(Norm(f, gauss), std::sqrt(Innerprod(f, f, gauss)));
  EXPECT_DOUBLE_EQ(Norm(g, gauss), std::sqrt(Innerprod(g, g, gauss)));
  // test normal frames
  Global normal = Global(0, 0, 1).normalized();
  for (int q = 0; q < gauss.CountPoints(); ++q) {
    auto &frame = gauss.GetNormalFrame(q);
    auto &nu = frame[0], &sigma = frame[1], &pi = frame[2];
    EXPECT_NEAR((nu - normal).norm(), 0.0, 1e-15);
    EXPECT_NEAR((nu - sigma.cross(pi)).norm(), 0.0, 1e-15);
    EXPECT_NEAR((sigma - pi.cross(nu)).norm(), 0.0, 1e-15);
    EXPECT_NEAR((pi - nu.cross(sigma)).norm(), 0.0, 1e-15);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
