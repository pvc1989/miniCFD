//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cmath>

#include "mini/gauss/function.hpp"
#include "mini/gauss/tetrahedron.hpp"
#include "mini/lagrange/tetrahedron.hpp"

#include "gtest/gtest.h"

class TestGaussTetrahedron : public ::testing::Test {
 protected:
  static constexpr int kPoints = 24;
  using Lagrange = mini::lagrange::Tetrahedron4<double>;
  using Gauss = mini::gauss::Tetrahedron<double, kPoints>;
  using Mat3x1 = mini::algebra::Matrix<double, 3, 1>;
};
TEST_F(TestGaussTetrahedron, OnStandardElement) {
  auto lagrange = Lagrange{
    Mat3x1(0, 0, 0), Mat3x1(3, 0, 0), Mat3x1(0, 3, 0), Mat3x1(0, 0, 3)
  };
  auto tetra = Gauss(lagrange);
  static_assert(tetra.CellDim() == 3);
  static_assert(tetra.PhysDim() == 3);
  EXPECT_NEAR(tetra.volume(), 4.5, 1e-14);
  EXPECT_EQ(tetra.center(), Mat3x1(0.75, 0.75, 0.75));
  EXPECT_EQ(tetra.CountPoints(), kPoints);
  EXPECT_EQ(tetra.LocalToGlobal(1, 0, 0), Mat3x1(0, 0, 0));
  EXPECT_EQ(tetra.LocalToGlobal(0, 1, 0), Mat3x1(3, 0, 0));
  EXPECT_EQ(tetra.LocalToGlobal(0, 0, 1), Mat3x1(0, 3, 0));
  EXPECT_EQ(tetra.LocalToGlobal(0, 0, 0), Mat3x1(0, 0, 3));
  EXPECT_EQ(tetra.GlobalToLocal(0, 0, 0), Mat3x1(1, 0, 0));
  EXPECT_EQ(tetra.GlobalToLocal(3, 0, 0), Mat3x1(0, 1, 0));
  EXPECT_EQ(tetra.GlobalToLocal(0, 3, 0), Mat3x1(0, 0, 1));
  EXPECT_EQ(tetra.GlobalToLocal(0, 0, 3), Mat3x1(0, 0, 0));
  EXPECT_NEAR(Quadrature([](Mat3x1 const&){ return 6.0; }, tetra), 1.0, 1e-14);
  EXPECT_NEAR(Integrate([](Mat3x1 const&){ return 6.0; }, tetra), 27.0, 1e-13);
  auto f = [](Mat3x1 const& xyz){ return xyz[0]; };
  auto g = [](Mat3x1 const& xyz){ return xyz[1]; };
  auto h = [](Mat3x1 const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, tetra), Integrate(h, tetra));
  EXPECT_DOUBLE_EQ(Norm(f, tetra), std::sqrt(Innerprod(f, f, tetra)));
  EXPECT_DOUBLE_EQ(Norm(g, tetra), std::sqrt(Innerprod(g, g, tetra)));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
