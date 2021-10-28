//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cmath>

#include "mini/integrator/function.hpp"
#include "mini/integrator/quad.hpp"
#include "mini/polynomial/basis.hpp"
#include "mini/polynomial/projection.hpp"

#include "gtest/gtest.h"

using std::sqrt;

class TestQuad4x4 : public ::testing::Test {
 protected:
  using Quad2D4x4 = mini::integrator::Quad<double, 2, 4, 4>;
  using Quad3D4x4 = mini::integrator::Quad<double, 3, 4, 4>;
  using Mat2x4 = mini::algebra::Matrix<double, 2, 4>;
  using Mat2x1 = mini::algebra::Matrix<double, 2, 1>;
  using Mat3x4 = mini::algebra::Matrix<double, 3, 4>;
  using Mat3x1 = mini::algebra::Matrix<double, 3, 1>;
  using Basis = mini::polynomial::OrthoNormal<double, 2, 2>;
  using Y = typename Basis::MatNx1;
  using A = typename Basis::MatNxN;
  using Mat1x6 = mini::algebra::Matrix<double, 1, 6>;
  using Mat7x1 = mini::algebra::Matrix<double, 7, 1>;
  using Mat7x6 = mini::algebra::Matrix<double, 7, 6>;
};
TEST_F(TestQuad4x4, OrthoNormal) {
  Mat2x1 origin = {0, 0}, left = {-1, 2}, right = {1, 3};
  Mat2x4 xyz_global_i;
  xyz_global_i.row(0) << -1, 1, 1, -1;
  xyz_global_i.row(1) << -1, -1, 1, 1;
  auto quad = Quad2D4x4(xyz_global_i);
  auto basis = Basis(quad);
  double residual = (Integrate([&basis](const Mat2x1& xy) {
    auto col = basis(xy);
    A prod = col * col.transpose();
    return prod;
  }, quad) - A::Identity()).cwiseAbs().maxCoeff();
  EXPECT_NEAR(residual, 0.0, 1e-15);
  auto x = left[0], y = left[1];
  xyz_global_i.row(0) << x-1, x+1, x+1, x-1;
  xyz_global_i.row(1) << y-1, y-1, y+1, y+1;
  quad = Quad2D4x4(xyz_global_i);
  basis = Basis(quad);
  residual = (Integrate([&basis](Mat2x1 const& xy) {
    auto col = basis(xy);
    A prod = col * col.transpose();
    return prod;
  }, quad) - A::Identity()).cwiseAbs().maxCoeff();
  EXPECT_NEAR(residual, 0.0, 1e-15);
}
TEST_F(TestQuad4x4, Projection) {
  Mat2x4 xyz_global_i;
  xyz_global_i.row(0) << -1, 1, 1, -1;
  xyz_global_i.row(1) << -1, -1, 1, 1;
  auto quad = Quad2D4x4(xyz_global_i);
  auto basis = Basis(quad);
  auto scalar_f = [](Mat2x1 const& xy){
    return xy[0] * xy[1];
  };
  using ScalarPF = mini::polynomial::Projection<double, 2, 2, 1>;
  auto scalar_pf = ScalarPF(scalar_f, basis);
  double residual = (scalar_pf.GetCoeff()
      - Mat1x6(0, 0, 0, 0, 1, 0)).cwiseAbs().maxCoeff();
  EXPECT_NEAR(residual, 0.0, 1e-15);
  auto vector_f = [](Mat2x1 const& xy) {
    auto x = xy[0], y = xy[1];
    Mat7x1 func(0, 1, x, y, x * x, x * y, y * y);
    return func;
  };
  using VectorPF = mini::polynomial::Projection<double, 2, 2, 7>;
  auto vector_pf = VectorPF(vector_f, basis);
  Mat7x6 exact_vector{
      {0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0},
      {0, 0, 1, 0, 0, 0}, {0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 1, 0},
      {0, 0, 0, 0, 0, 1}
  };
  Mat7x6 abs_diff = vector_pf.GetCoeff() - exact_vector;
  EXPECT_NEAR(abs_diff.cwiseAbs().maxCoeff(), 0.0, 1e-15);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
