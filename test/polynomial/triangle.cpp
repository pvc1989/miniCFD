//  Copyright 2022 PEI Weicheng

#include <cmath>

#include "mini/gauss/function.hpp"
#include "mini/gauss/triangle.hpp"
#include "mini/polynomial/basis.hpp"
#include "mini/polynomial/projection.hpp"

#include "gtest/gtest.h"

using std::sqrt;

class TestTriangle : public ::testing::Test {
 protected:
  using Triangle = mini::gauss::Triangle<double, 2, 12>;
  using Mat2x3 = mini::algebra::Matrix<double, 2, 3>;
  using Basis = mini::polynomial::OrthoNormal<double, 2, 2>;
  using Coord = typename Basis::Coord;
  using A = typename Basis::MatNxN;
};
TEST_F(TestTriangle, OrthoNormal) {
  // build a triangle-gauss
  Mat2x3 xy_global_i;
  xy_global_i.row(0) << 10, 0, 0;
  xy_global_i.row(1) << 0, 10, 0;
  auto triangle = Triangle(xy_global_i);
  // build an orthonormal basis on it
  auto basis = Basis(triangle);
  // check orthonormality
  double residual = (Integrate([&basis](const Coord& xyz) {
    auto col = basis(xyz);
    A prod = col * col.transpose();
    return prod;
  }, triangle) - A::Identity()).cwiseAbs().maxCoeff();
  EXPECT_NEAR(residual, 0.0, 1e-14);
  // build another triangle-gauss
  Coord shift = {10, 20};
  xy_global_i.col(0) += shift;
  xy_global_i.col(1) += shift;
  triangle = Triangle(xy_global_i);
  // build another orthonormal basis on it
  basis = Basis(triangle);
  // check orthonormality
  residual = (Integrate([&basis](const Coord& xyz) {
    auto col = basis(xyz);
    A prod = col * col.transpose();
    return prod;
  }, triangle) - A::Identity()).cwiseAbs().maxCoeff();
  EXPECT_NEAR(residual, 0.0, 1e-14);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
