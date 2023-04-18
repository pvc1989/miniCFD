//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cmath>

#include "mini/integrator/function.hpp"
#include "mini/integrator/hexahedron.hpp"

#include "gtest/gtest.h"

class TestHexahedronIntegrator : public ::testing::Test {
 protected:
  using Hexahedron = mini::integrator::Hexahedron<double, 4, 4, 4>;
  using Mat1x8 = mini::algebra::Matrix<double, 1, 8>;
  using Mat3x8 = mini::algebra::Matrix<double, 3, 8>;
  using Mat3x1 = mini::algebra::Matrix<double, 3, 1>;
};
TEST_F(TestHexahedronIntegrator, VirtualMethods) {
  Mat3x8 xyz_global_i;
  xyz_global_i.row(0) << -1, +1, +1, -1, -1, +1, +1, -1;
  xyz_global_i.row(1) << -1, -1, +1, +1, -1, -1, +1, +1;
  xyz_global_i.row(2) << -1, -1, -1, -1, +1, +1, +1, +1;
  auto hexa = Hexahedron(xyz_global_i);
  static_assert(hexa.CellDim() == 3);
  static_assert(hexa.PhysDim() == 3);
  EXPECT_NEAR(hexa.volume(), 8.0, 1e-14);
  EXPECT_EQ(hexa.CountQuadraturePoints(), 64);
  auto p0 = hexa.GetLocalCoord(0);
  EXPECT_EQ(p0[0], -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7));
  EXPECT_EQ(p0[1], -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7));
  EXPECT_EQ(p0[2], -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7));
  auto w1d = (18 - std::sqrt(30)) / 36.0;
  EXPECT_EQ(hexa.GetLocalWeight(0), w1d * w1d * w1d);
}
TEST_F(TestHexahedronIntegrator, CommonMethods) {
  Mat3x8 xyz_global_i;
  xyz_global_i.row(0) << -1, +1, +1, -1, -1, +1, +1, -1;
  xyz_global_i.row(1) << -1, -1, +1, +1, -1, -1, +1, +1;
  xyz_global_i.row(2) << -1, -1, -1, -1, +1, +1, +1, +1;
  xyz_global_i.array() *= 10.0;
  auto hexa = Hexahedron(xyz_global_i);
  EXPECT_EQ(hexa.LocalToGlobal(1, 1, 1), Mat3x1(10, 10, 10));
  EXPECT_EQ(hexa.LocalToGlobal(1.5, 1.5, 1.5), Mat3x1(15, 15, 15));
  EXPECT_EQ(hexa.LocalToGlobal(3, 4, 5), Mat3x1(30, 40, 50));
  EXPECT_EQ(hexa.global_to_local_3x1(30, 40, 20), Mat3x1(3, 4, 2));
  EXPECT_EQ(hexa.global_to_local_3x1(40, 55, 25), Mat3x1(4, 5.5, 2.5));
  EXPECT_EQ(hexa.global_to_local_3x1(70, 130, 60), Mat3x1(7, 13, 6));
  EXPECT_DOUBLE_EQ(Quadrature([](Mat3x1 const&){ return 2.0; }, hexa), 16.0);
  EXPECT_NEAR(Integrate([](Mat3x1 const&){ return 2.0; }, hexa), 16000, 1e-10);
  auto f = [](Mat3x1 const& xyz){ return xyz[0]; };
  auto g = [](Mat3x1 const& xyz){ return xyz[1]; };
  auto h = [](Mat3x1 const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, hexa), Integrate(h, hexa));
  EXPECT_DOUBLE_EQ(Norm(f, hexa), std::sqrt(Innerprod(f, f, hexa)));
  EXPECT_DOUBLE_EQ(Norm(g, hexa), std::sqrt(Innerprod(g, g, hexa)));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
