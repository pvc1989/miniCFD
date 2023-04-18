//  Copyright 2022 PEI Weicheng

#include <cmath>

#include "mini/integrator/function.hpp"
#include "mini/integrator/tetrahedron.hpp"
#include "mini/polynomial/basis.hpp"
#include "mini/polynomial/projection.hpp"

#include "gtest/gtest.h"

using std::sqrt;

class TestTetrahedron : public ::testing::Test {
 protected:
  using Tetrahedron = mini::integrator::Tetrahedron<double, 14>;
  using Mat3x4 = mini::algebra::Matrix<double, 3, 4>;
  using Basis = mini::polynomial::OrthoNormal<double, 3, 2>;
  using Coord = typename Basis::Coord;
  using A = typename Basis::MatNxN;
};
TEST_F(TestTetrahedron, OrthoNormal) {
  // build a tetra-integrator
  Mat3x4 xyz_global_i;
  xyz_global_i.row(0) << 10, 0, 10, 10;
  xyz_global_i.row(1) << 10, 10, 0, 10;
  xyz_global_i.row(2) << 0, 10, 10, 10;
  auto tetra = Tetrahedron(xyz_global_i);
  // build an orthonormal basis on it
  auto basis = Basis(tetra);
  // check orthonormality
  double residual = (Integrate([&basis](const Coord& xyz) {
    auto col = basis(xyz);
    A prod = col * col.transpose();
    return prod;
  }, tetra) - A::Identity()).cwiseAbs().maxCoeff();
  EXPECT_NEAR(residual, 0.0, 1e-14);
  // build another tetra-integrator
  Coord shift = {10, 20, 30};
  xyz_global_i.col(0) += shift;
  xyz_global_i.col(1) += shift;
  xyz_global_i.col(2) += shift;
  xyz_global_i.col(3) += shift;
  tetra = Tetrahedron(xyz_global_i);
  // build another orthonormal basis on it
  basis = Basis(tetra);
  // check orthonormality
  residual = (Integrate([&basis](const Coord& xyz) {
    auto col = basis(xyz);
    A prod = col * col.transpose();
    return prod;
  }, tetra) - A::Identity()).cwiseAbs().maxCoeff();
  EXPECT_NEAR(residual, 0.0, 1e-14);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
