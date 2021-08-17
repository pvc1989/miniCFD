//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cmath>

#include "mini/integrator/base.hpp"
#include "mini/integrator/quad.hpp"

#include "gtest/gtest.h"

using std::sqrt;

namespace mini {
namespace integrator {

class TestQuad4x4 : public ::testing::Test {
 protected:
  using Quad4x4_2d = Quad<double, 4, 4, 2>;
  using Quad4x4_3d = Quad<double, 4, 4, 3>;
  using Mat2x4 = Eigen::Matrix<double, 2, 4>;
  using Mat2x1 = Eigen::Matrix<double, 2, 1>;
  using Mat3x4 = Eigen::Matrix<double, 3, 4>;
  using Mat3x1 = Eigen::Matrix<double, 3, 1>;
  using B = Basis<double, 2, 2>;
  using Y = typename B::MatNx1;
  using A = typename B::MatNxN;
};
TEST_F(TestQuad4x4, StaticMethods) {
  static_assert(Quad4x4_2d::CountQuadPoints() == 16);
  static_assert(Quad4x4_2d::CellDim() == 2);
  static_assert(Quad4x4_2d::PhysDim() == 2);
  auto p0 = Quad4x4_2d::GetCoord(0);
  EXPECT_EQ(p0[0], -std::sqrt((3 - 2 * std::sqrt(1.2)) / 7));
  EXPECT_EQ(p0[1], -std::sqrt((3 - 2 * std::sqrt(1.2)) / 7));
  auto w1d = (18 + std::sqrt(30)) / 36.0;
  EXPECT_EQ(Quad4x4_2d::GetWeight(0), w1d * w1d);
}
TEST_F(TestQuad4x4, In2dSpace) {
  static_assert(Quad4x4_2d::PhysDim() == 2);
  Mat2x4 xyz_global_i;
  xyz_global_i.row(0) << -1, 1, 1, -1;
  xyz_global_i.row(1) << -1, -1, 1, 1;
  auto quad = Quad4x4_2d(xyz_global_i);
  EXPECT_EQ(quad.local_to_global_Dx1(0, 0), Mat2x1(0, 0));
  EXPECT_EQ(quad.local_to_global_Dx1(1, 1), Mat2x1(1, 1));
  EXPECT_EQ(quad.local_to_global_Dx1(-1, -1), Mat2x1(-1, -1));
  EXPECT_DOUBLE_EQ(Quadrature([](Mat2x1){ return 2.0; }, quad), 8.0);
  EXPECT_DOUBLE_EQ(Integrate([](Mat2x1){ return 2.0; }, quad), 8.0);
  auto f = [](Mat2x1 xy){ return xy[0]; };
  auto g = [](Mat2x1 xy){ return xy[1]; };
  auto h = [](Mat2x1 xy){ return xy[0] * xy[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, quad), Integrate(h, quad));
  EXPECT_DOUBLE_EQ(Norm(f, quad), sqrt(Innerprod(f, f, quad)));
  EXPECT_DOUBLE_EQ(Norm(g, quad), sqrt(Innerprod(g, g, quad)));
}
TEST_F(TestQuad4x4, In3dSpace) {
  static_assert(Quad4x4_3d::PhysDim() == 3);
  Mat3x4 xyz_global_i;
  xyz_global_i.row(0) << -1, 1, 1, -1;
  xyz_global_i.row(1) << -1, -1, 1, 1;
  xyz_global_i.row(2) << -1, -1, 1, 1;
  auto quad = Quad4x4_3d(xyz_global_i);
  EXPECT_EQ(quad.local_to_global_Dx1(0, 0), Mat3x1(0, 0, 0));
  EXPECT_EQ(quad.local_to_global_Dx1(1, 1), Mat3x1(1, 1, 1));
  EXPECT_EQ(quad.local_to_global_Dx1(-1, -1), Mat3x1(-1, -1, -1));
  EXPECT_DOUBLE_EQ(Quadrature([](Mat2x1){ return 2.0; }, quad), 8.0);
  EXPECT_DOUBLE_EQ(Integrate([](Mat3x1){ return 2.0; }, quad), sqrt(2) * 8.0);
  auto f = [](Mat3x1 xyz){ return xyz[0]; };
  auto g = [](Mat3x1 xyz){ return xyz[1]; };
  auto h = [](Mat3x1 xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, quad), Integrate(h, quad));
  EXPECT_DOUBLE_EQ(Norm(f, quad), sqrt(Innerprod(f, f, quad)));
  EXPECT_DOUBLE_EQ(Norm(g, quad), sqrt(Innerprod(g, g, quad)));
}
TEST_F(TestQuad4x4, Basis) {
  Mat2x1 origin = {0, 0}, left = {-1, 2}, right = {1, 3};
  auto b = B();
  EXPECT_EQ(b(origin), Y(1, 0, 0, 0, 0, 0));
  EXPECT_EQ(b(left), Y(1, -1, 2, 1, -2, 4));
  EXPECT_EQ(b(right), Y(1, 1, 3, 1, 3, 9));
  Mat2x4 xyz_global_i;
  xyz_global_i.row(0) << -1, 1, 1, -1;
  xyz_global_i.row(1) << -1, -1, 1, 1;
  auto quad = Quad4x4_2d(xyz_global_i);
  Orthonormalize(&b, quad);
  auto gram = [&b](const Mat2x1& xy) {
    auto col = b(xy);
    A prod = col * col.transpose();
    return prod;
  };
  print(Integrate(gram, quad));
  b = B(left);
  EXPECT_EQ(b(origin), Y(1, 1, -2, 1, -2, 4));
  EXPECT_EQ(b(left), Y(1, 0, 0, 0, 0, 0));
  EXPECT_EQ(b(right), Y(1, 2, 1, 4, 2, 1));
}

}  // namespace integrator
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
