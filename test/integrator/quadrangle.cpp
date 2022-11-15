//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cmath>

#include "mini/integrator/function.hpp"
#include "mini/integrator/quadrangle.hpp"

#include "gtest/gtest.h"

class TestQuadrangleIntegrator : public ::testing::Test {
 protected:
  using Quadrangle2D = mini::integrator::Quadrangle<double, 2, 4, 4>;
  using Quadrangle3D = mini::integrator::Quadrangle<double, 3, 4, 4>;
  using Mat2x4 = mini::algebra::Matrix<double, 2, 4>;
  using Mat2x1 = mini::algebra::Matrix<double, 2, 1>;
  using Mat3x4 = mini::algebra::Matrix<double, 3, 4>;
  using Mat3x1 = mini::algebra::Matrix<double, 3, 1>;
};
TEST_F(TestQuadrangleIntegrator, VirtualMethods) {
  Mat2x4 xyz_global_i;
  xyz_global_i.row(0) << -1, 1, 1, -1;
  xyz_global_i.row(1) << -1, -1, 1, 1;
  auto quad = Quadrangle2D(xyz_global_i);
  EXPECT_EQ(quad.CountQuadraturePoints(), 16);
  auto p0 = quad.GetLocalCoord(0);
  EXPECT_EQ(p0[0], -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7));
  EXPECT_EQ(p0[1], -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7));
  auto w1d = (18 - std::sqrt(30)) / 36.0;
  EXPECT_EQ(quad.GetLocalWeight(0), w1d * w1d);
}
TEST_F(TestQuadrangleIntegrator, In2dSpace) {
  Mat2x4 xyz_global_i;
  xyz_global_i.row(0) << -1, 1, 1, -1;
  xyz_global_i.row(1) << -1, -1, 1, 1;
  auto quad = Quadrangle2D(xyz_global_i);
  static_assert(quad.CellDim() == 2);
  static_assert(quad.PhysDim() == 2);
  EXPECT_NEAR(quad.area(), 4.0, 1e-15);
  EXPECT_EQ(quad.LocalToGlobal(0, 0), Mat2x1(0, 0));
  EXPECT_EQ(quad.LocalToGlobal(1, 1), Mat2x1(1, 1));
  EXPECT_EQ(quad.LocalToGlobal(-1, -1), Mat2x1(-1, -1));
  EXPECT_DOUBLE_EQ(Quadrature([](Mat2x1 const&){ return 2.0; }, quad), 8.0);
  EXPECT_DOUBLE_EQ(Integrate([](Mat2x1 const&){ return 2.0; }, quad), 8.0);
  auto f = [](Mat2x1 const& xy){ return xy[0]; };
  auto g = [](Mat2x1 const& xy){ return xy[1]; };
  auto h = [](Mat2x1 const& xy){ return xy[0] * xy[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, quad), Integrate(h, quad));
  EXPECT_DOUBLE_EQ(Norm(f, quad), std::sqrt(Innerprod(f, f, quad)));
  EXPECT_DOUBLE_EQ(Norm(g, quad), std::sqrt(Innerprod(g, g, quad)));
}
TEST_F(TestQuadrangleIntegrator, In3dSpace) {
  Mat3x4 xyz_global_i;
  xyz_global_i.row(0) << 0, 4, 4, 0;
  xyz_global_i.row(1) << 0, 0, 4, 4;
  xyz_global_i.row(2) << 0, 0, 4, 4;
  auto quad = Quadrangle3D(xyz_global_i);
  static_assert(quad.CellDim() == 2);
  static_assert(quad.PhysDim() == 3);
  EXPECT_NEAR(quad.area(), sqrt(2) * 16.0, 1e-14);
  EXPECT_EQ(quad.LocalToGlobal(0, 0), Mat3x1(2, 2, 2));
  EXPECT_EQ(quad.LocalToGlobal(+1, +1), Mat3x1(4, 4, 4));
  EXPECT_EQ(quad.LocalToGlobal(-1, -1), Mat3x1(0, 0, 0));
  EXPECT_DOUBLE_EQ(
      Quadrature([](Mat2x1 const&){ return 2.0; }, quad), 8.0);
  EXPECT_DOUBLE_EQ(
      Integrate([](Mat3x1 const&){ return 2.0; }, quad), sqrt(2) * 32.0);
  auto f = [](Mat3x1 const& xyz){ return xyz[0]; };
  auto g = [](Mat3x1 const& xyz){ return xyz[1]; };
  auto h = [](Mat3x1 const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, quad), Integrate(h, quad));
  EXPECT_DOUBLE_EQ(Norm(f, quad), std::sqrt(Innerprod(f, f, quad)));
  EXPECT_DOUBLE_EQ(Norm(g, quad), std::sqrt(Innerprod(g, g, quad)));
  // test normal frames
  quad.BuildNormalFrames();
  for (int q = 0; q < quad.CountQuadraturePoints(); ++q) {
    auto &frame = quad.GetNormalFrame(q);
    auto &nu = frame.col(0), &sigma = frame.col(1), &pi = frame.col(2);
    EXPECT_NEAR((nu - sigma.cross(pi)).cwiseAbs().maxCoeff(), 0.0, 1e-15);
    EXPECT_NEAR((sigma - pi.cross(nu)).cwiseAbs().maxCoeff(), 0.0, 1e-15);
    EXPECT_NEAR((pi - nu.cross(sigma)).cwiseAbs().maxCoeff(), 0.0, 1e-15);
    EXPECT_NEAR((sigma - Mat3x1(1, 0, 0)).cwiseAbs().maxCoeff(), 0.0, 1e-15);
    auto vec = Mat3x1(0, +std::sqrt(0.5), std::sqrt(0.5));
    EXPECT_NEAR((pi - vec).cwiseAbs().maxCoeff(), 0.0, 1e-15);
    vec[1] *= -1;
    EXPECT_NEAR((nu - vec).cwiseAbs().maxCoeff(), 0.0, 1e-15);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
