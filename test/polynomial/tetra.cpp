//  Copyright 2022 PEI Weicheng

#include <cmath>

#include "mini/integrator/function.hpp"
#include "mini/integrator/tetra.hpp"
#include "mini/polynomial/basis.hpp"
#include "mini/polynomial/projection.hpp"

#include "gtest/gtest.h"

using std::sqrt;

class TestTetra : public ::testing::Test {
 protected:
  using Tetra = mini::integrator::Tetra<double, 24>;
  using Mat3x4 = mini::algebra::Matrix<double, 3, 4>;
  using Coord = mini::algebra::Matrix<double, 3, 1>;
  using Basis = mini::polynomial::OrthoNormal<double, 3, 2>;
  using A = typename Basis::MatNxN;
};
TEST_F(TestTetra, OrthoNormal) {
  // build a tetra-integrator
  Mat3x4 xyz_global_i;
  xyz_global_i.row(0) << 1, 0, 1, 1;
  xyz_global_i.row(1) << 1, 1, 0, 1;
  xyz_global_i.row(2) << 0, 1, 1, 1;
  auto tetra = Tetra(xyz_global_i);
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
  Coord shift = {-1, 2, 3};
  xyz_global_i.col(0) += shift;
  xyz_global_i.col(1) += shift;
  xyz_global_i.col(2) += shift;
  xyz_global_i.col(3) += shift;
  tetra = Tetra(xyz_global_i);
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
