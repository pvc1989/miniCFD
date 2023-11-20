//  Copyright 2022 PEI Weicheng

#include <cmath>

#include "mini/gauss/function.hpp"
#include "mini/gauss/triangle.hpp"
#include "mini/geometry/triangle.hpp"
#include "mini/basis/linear.hpp"
#include "mini/polynomial/projection.hpp"

#include "gtest/gtest.h"

using std::sqrt;

class TestTriangle : public ::testing::Test {
};
TEST_F(TestTriangle, OrthoNormal) {
  using Basis = mini::basis::OrthoNormal<double, 2, 2>;
  using Gauss = mini::gauss::Triangle<double, 2, 12>;
  using Lagrange = mini::geometry::Triangle3<double, 2>;
  using Coord = typename Lagrange::Global;
  // build a triangle-gauss
  auto lagrange = Lagrange { Coord(10, 0), Coord(0, 10), Coord(0, 0) };
  auto gauss = Gauss(lagrange);
  // build an orthonormal basis on it
  auto basis = Basis(gauss);
  // check orthonormality
  using A = typename Basis::MatNxN;
  double residual = (Integrate([&basis](const Coord& xyz) {
    auto col = basis(xyz);
    A prod = col * col.transpose();
    return prod;
  }, gauss) - A::Identity()).norm();
  EXPECT_NEAR(residual, 0.0, 1e-14);
  // build another triangle-gauss
  Coord shift = {10, 20};
  lagrange = Lagrange {
    Coord(10, 0) + shift, Coord(0, 10) + shift, Coord(0, 0) + shift
  };
  gauss = Gauss(lagrange);
  // build another orthonormal basis on it
  basis = Basis(gauss);
  // check orthonormality
  residual = (Integrate([&basis](const Coord& xyz) {
    auto col = basis(xyz);
    A prod = col * col.transpose();
    return prod;
  }, gauss) - A::Identity()).norm();
  EXPECT_NEAR(residual, 0.0, 1e-14);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
