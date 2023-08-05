//  Copyright 2023 PEI Weicheng

#include <cmath>

#include "mini/gauss/function.hpp"
#include "mini/gauss/pyramid.hpp"
#include "mini/lagrange/pyramid.hpp"

#include "gtest/gtest.h"

class TestGaussPyramid : public ::testing::Test {
 protected:
  using Lagrange = mini::lagrange::Pyramid5<double>;
  using Gauss = mini::gauss::Pyramid<double, 4, 4, 3>;
  using Coord = mini::algebra::Matrix<double, 3, 1>;
};
TEST_F(TestGaussPyramid, OnStandardElement) {
  auto a = 2.0, b = 3.0, h = 4.0;
  auto lagrange = Lagrange{
    Coord(-a, -b, 0), Coord(+a, -b, 0),
    Coord(+a, +b, 0), Coord(-a, +b, 0),
    Coord(0, 0, h)
  };
  auto gauss = Gauss(lagrange);
  static_assert(gauss.CellDim() == 3);
  static_assert(gauss.PhysDim() == 3);
  auto volume = (a + a) * (b + b) * h / 3;
  EXPECT_NEAR(gauss.volume(), volume, 1e-13);
  EXPECT_EQ(gauss.CountPoints(), 48);
  EXPECT_NEAR(Integrate([](Coord const&){ return 2.0; }, gauss),
      volume * 2, 1e-13);
  auto f = [](Coord const& xyz){ return xyz[0]; };
  auto g = [](Coord const& xyz){ return xyz[1]; };
  auto fg = [](Coord const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, gauss), Integrate(fg, gauss));
  EXPECT_DOUBLE_EQ(Norm(f, gauss), std::sqrt(Innerprod(f, f, gauss)));
  EXPECT_DOUBLE_EQ(Norm(g, gauss), std::sqrt(Innerprod(g, g, gauss)));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
