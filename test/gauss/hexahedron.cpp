//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cmath>

#include "mini/gauss/function.hpp"
#include "mini/gauss/hexahedron.hpp"
#include "mini/lagrange/hexahedron.hpp"

#include "gtest/gtest.h"

class TestGaussHexahedron : public ::testing::Test {
 protected:
  using Gauss = mini::gauss::Hexahedron<double, 4, 4, 4>;
  using Mat3x1 = mini::algebra::Matrix<double, 3, 1>;
  using Lagrange = mini::lagrange::Hexahedron8<double>;
  
};
TEST_F(TestGaussHexahedron, OnStandardElement) {
  auto lagrange = Lagrange {
    Mat3x1(-1, -1, -1), Mat3x1(+1, -1, -1),
    Mat3x1(+1, +1, -1), Mat3x1(-1, +1, -1),
    Mat3x1(-1, -1, +1), Mat3x1(+1, -1, +1),
    Mat3x1(+1, +1, +1), Mat3x1(-1, +1, +1)
  };
  auto hexa = Gauss(lagrange);
  static_assert(hexa.CellDim() == 3);
  static_assert(hexa.PhysDim() == 3);
  EXPECT_NEAR(hexa.volume(), 8.0, 1e-14);
  EXPECT_EQ(hexa.CountPoints(), 64);
  auto p0 = hexa.GetLocalCoord(0);
  EXPECT_EQ(p0[0], -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7));
  EXPECT_EQ(p0[1], -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7));
  EXPECT_EQ(p0[2], -std::sqrt((3 + 2 * std::sqrt(1.2)) / 7));
  auto w1d = (18 - std::sqrt(30)) / 36.0;
  EXPECT_EQ(hexa.GetLocalWeight(0), w1d * w1d * w1d);
}
TEST_F(TestGaussHexahedron, OnScaledElement) {
  auto lagrange = Lagrange {
    Mat3x1(-10, -10, -10), Mat3x1(+10, -10, -10),
    Mat3x1(+10, +10, -10), Mat3x1(-10, +10, -10),
    Mat3x1(-10, -10, +10), Mat3x1(+10, -10, +10),
    Mat3x1(+10, +10, +10), Mat3x1(-10, +10, +10)
  };
  auto hexa = Gauss(lagrange);
  EXPECT_EQ(hexa.LocalToGlobal(1, 1, 1), Mat3x1(10, 10, 10));
  EXPECT_EQ(hexa.LocalToGlobal(1.5, 1.5, 1.5), Mat3x1(15, 15, 15));
  EXPECT_EQ(hexa.LocalToGlobal(3, 4, 5), Mat3x1(30, 40, 50));
  EXPECT_EQ(hexa.GlobalToLocal(30, 40, 20), Mat3x1(3, 4, 2));
  EXPECT_EQ(hexa.GlobalToLocal(40, 55, 25), Mat3x1(4, 5.5, 2.5));
  EXPECT_EQ(hexa.GlobalToLocal(70, 130, 60), Mat3x1(7, 13, 6));
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
