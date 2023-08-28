//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cmath>

#include "mini/gauss/function.hpp"
#include "mini/gauss/tetrahedron.hpp"
#include "mini/lagrange/tetrahedron.hpp"

#include "gtest/gtest.h"

class TestGaussTetrahedron : public ::testing::Test {
 protected:
  static constexpr int kPoints = 24;
  using Gauss = mini::gauss::Tetrahedron<double, kPoints>;
  using Coord = typename Gauss::Global;
};
TEST_F(TestGaussTetrahedron, OnLinearElement) {
  using Lagrange = mini::lagrange::Tetrahedron4<double>;
  auto lagrange = Lagrange{
    Coord(0, 0, 0), Coord(3, 0, 0), Coord(0, 3, 0), Coord(0, 0, 3)
  };
  auto tetra = Gauss(lagrange);
  static_assert(tetra.CellDim() == 3);
  static_assert(tetra.PhysDim() == 3);
  EXPECT_NEAR(tetra.volume(), 4.5, 1e-14);
  EXPECT_EQ(tetra.center(), Coord(0.75, 0.75, 0.75));
  EXPECT_EQ(tetra.CountPoints(), kPoints);
  EXPECT_EQ(tetra.LocalToGlobal(1, 0, 0), Coord(0, 0, 0));
  EXPECT_EQ(tetra.LocalToGlobal(0, 1, 0), Coord(3, 0, 0));
  EXPECT_EQ(tetra.LocalToGlobal(0, 0, 1), Coord(0, 3, 0));
  EXPECT_EQ(tetra.LocalToGlobal(0, 0, 0), Coord(0, 0, 3));
  EXPECT_EQ(tetra.GlobalToLocal(0, 0, 0), Coord(1, 0, 0));
  EXPECT_EQ(tetra.GlobalToLocal(3, 0, 0), Coord(0, 1, 0));
  EXPECT_EQ(tetra.GlobalToLocal(0, 3, 0), Coord(0, 0, 1));
  EXPECT_EQ(tetra.GlobalToLocal(0, 0, 3), Coord(0, 0, 0));
  EXPECT_NEAR(Quadrature([](Coord const&){ return 6.0; }, tetra), 1.0, 1e-14);
  EXPECT_NEAR(Integrate([](Coord const&){ return 6.0; }, tetra), 27.0, 1e-13);
  auto f = [](Coord const& xyz){ return xyz[0]; };
  auto g = [](Coord const& xyz){ return xyz[1]; };
  auto h = [](Coord const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, tetra), Integrate(h, tetra));
  EXPECT_DOUBLE_EQ(Norm(f, tetra), std::sqrt(Innerprod(f, f, tetra)));
  EXPECT_DOUBLE_EQ(Norm(g, tetra), std::sqrt(Innerprod(g, g, tetra)));
}
TEST_F(TestGaussTetrahedron, OnQuadraticElement) {
  using Lagrange = mini::lagrange::Tetrahedron10<double>;
  double a = 1.5;
  auto lagrange = Lagrange{
    Coord(0, 0, 0), Coord(a*2, 0, 0), Coord(0, a*2, 0), Coord(0, 0, a*2),
    Coord(a, 0, 0), Coord(a, a, 0), Coord(0, a, 0),
    Coord(0, 0, a), Coord(a, 0, a), Coord(0, a, a),
  };
  auto tetra = Gauss(lagrange);
  static_assert(tetra.CellDim() == 3);
  static_assert(tetra.PhysDim() == 3);
  EXPECT_NEAR(tetra.volume(), 4.5, 1e-14);
  EXPECT_EQ(tetra.center(), Coord(0.75, 0.75, 0.75));
  EXPECT_EQ(tetra.CountPoints(), kPoints);
  EXPECT_EQ(tetra.LocalToGlobal(1, 0, 0), Coord(0, 0, 0));
  EXPECT_EQ(tetra.LocalToGlobal(0, 1, 0), Coord(3, 0, 0));
  EXPECT_EQ(tetra.LocalToGlobal(0, 0, 1), Coord(0, 3, 0));
  EXPECT_EQ(tetra.LocalToGlobal(0, 0, 0), Coord(0, 0, 3));
  EXPECT_EQ(tetra.GlobalToLocal(0, 0, 0), Coord(1, 0, 0));
  EXPECT_EQ(tetra.GlobalToLocal(3, 0, 0), Coord(0, 1, 0));
  EXPECT_EQ(tetra.GlobalToLocal(0, 3, 0), Coord(0, 0, 1));
  EXPECT_EQ(tetra.GlobalToLocal(0, 0, 3), Coord(0, 0, 0));
  EXPECT_NEAR(Quadrature([](Coord const&){ return 6.0; }, tetra), 1.0, 1e-14);
  EXPECT_NEAR(Integrate([](Coord const&){ return 6.0; }, tetra), 27.0, 1e-13);
  auto f = [](Coord const& xyz){ return xyz[0]; };
  auto g = [](Coord const& xyz){ return xyz[1]; };
  auto h = [](Coord const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, tetra), Integrate(h, tetra));
  EXPECT_DOUBLE_EQ(Norm(f, tetra), std::sqrt(Innerprod(f, f, tetra)));
  EXPECT_DOUBLE_EQ(Norm(g, tetra), std::sqrt(Innerprod(g, g, tetra)));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
