//  Copyright 2022 PEI Weicheng

#include <cmath>

#include "mini/gauss/function.hpp"
#include "mini/gauss/tetrahedron.hpp"
#include "mini/lagrange/tetrahedron.hpp"
#include "mini/basis/linear.hpp"
#include "mini/polynomial/projection.hpp"

#include "gtest/gtest.h"

using std::sqrt;

class TestTetrahedron : public ::testing::Test {
 protected:
  using Gauss = mini::gauss::Tetrahedron<double, 14>;
  using Lagrange = mini::lagrange::Tetrahedron4<double>;
  using Basis = mini::basis::OrthoNormal<double, 3, 2>;
  using Coord = typename Basis::Coord;
  using A = typename Basis::MatNxN;
};
TEST_F(TestTetrahedron, OrthoNormal) {
  // build a tetra-gauss
  auto lagrange = Lagrange(
    Coord(10, 10, 0), Coord(0, 10, 10),
    Coord(10, 0, 10), Coord(10, 10, 10)
  );
  auto tetra = Gauss(lagrange);
  // build an orthonormal basis on it
  auto basis = Basis(tetra);
  // check orthonormality
  double residual = (Integrate([&basis](const Coord& xyz) {
    auto col = basis(xyz);
    A prod = col * col.transpose();
    return prod;
  }, tetra) - A::Identity()).norm();
  EXPECT_NEAR(residual, 0.0, 1e-14);
  // build another tetra-gauss
  Coord shift = {10, 20, 30};
  lagrange = Lagrange(
    lagrange.GetGlobalCoord(0) + shift,
    lagrange.GetGlobalCoord(1) + shift,
    lagrange.GetGlobalCoord(2) + shift,
    lagrange.GetGlobalCoord(3) + shift
  );
  tetra = Gauss(lagrange);
  // build another orthonormal basis on it
  basis = Basis(tetra);
  // check orthonormality
  residual = (Integrate([&basis](const Coord& xyz) {
    auto col = basis(xyz);
    A prod = col * col.transpose();
    return prod;
  }, tetra) - A::Identity()).norm();
  EXPECT_NEAR(residual, 0.0, 1e-14);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
