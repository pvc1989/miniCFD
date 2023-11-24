// Copyright 2021 PEI WeiCheng and JIANG YuYan

#include "gtest/gtest.h"

#include "mini/riemann/euler/eigen.hpp"

namespace mini {
namespace riemann {
namespace euler {

class TestEigenMatrices : public ::testing::Test {
};
TEST_F(TestEigenMatrices, OrthoNormality) {
  auto rho{0.1}, u{0.2}, v{0.3}, w{0.4}, p{0.5};
  auto state = Primitives<double, 3>{rho, u, v, w, p};
  using Matrices = EigenMatrices<IdealGas<double, 1.4>>;
  typename Matrices::Mat3x1 nu{ 0, 0, 1 }, sigma{ 1, 0, 0 }, pi{ 0, 1, 0 };
  auto eigen_matrices = Matrices(state, nu, sigma, pi);
  auto &L = eigen_matrices.L;
  auto &R = eigen_matrices.R;
  auto &I = decltype(eigen_matrices.L)::Identity();
  EXPECT_NEAR((R * L - I).norm(), 0, 1e-15);
  EXPECT_NEAR((L * R - I).norm(), 0, 1e-15);
}

}  // namespace euler
}  // namespace riemann
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
