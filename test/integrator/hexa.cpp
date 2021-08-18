//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cmath>

#include "mini/integrator/base.hpp"
#include "mini/integrator/hexa.hpp"

#include "gtest/gtest.h"

using std::sqrt;

namespace mini {
namespace integrator {

class TestHexa4x4x4 : public ::testing::Test {
 protected:
  using Hexa4x4x4 = Hexa<double, 4, 4, 4>;
  using Mat1x8 = Eigen::Matrix<double, 1, 8>;
  using Mat3x8 = Eigen::Matrix<double, 3, 8>;
  using Mat3x1 = Eigen::Matrix<double, 3, 1>;
  using B = Basis<double, 3, 2>;
  using Y = typename B::MatNx1;
  using A = typename B::MatNxN;
  using Pscalar = ProjFunc<double, 3, 2, 1>;
  using Mat1x10 = Eigen::Matrix<double, 1, 10>;
  using Pvector = ProjFunc<double, 3, 2, 11>;
  using Mat11x1 = Eigen::Matrix<double, 11, 1>;
  using Mat11x10 = Eigen::Matrix<double, 11, 10>;
};
TEST_F(TestHexa4x4x4, StaticMethods) {
  static_assert(Hexa4x4x4::CountQuadPoints() == 64);
  static_assert(Hexa4x4x4::CellDim() == 3);
  static_assert(Hexa4x4x4::PhysDim() == 3);
  auto p0 = Hexa4x4x4::GetCoord(0);
  EXPECT_EQ(p0[0], -std::sqrt((3 - 2 * std::sqrt(1.2)) / 7));
  EXPECT_EQ(p0[1], -std::sqrt((3 - 2 * std::sqrt(1.2)) / 7));
  EXPECT_EQ(p0[2], -std::sqrt((3 - 2 * std::sqrt(1.2)) / 7));
  auto w1d = (18 + std::sqrt(30)) / 36.0;
  EXPECT_EQ(Hexa4x4x4::GetWeight(0), w1d * w1d * w1d);
}
TEST_F(TestHexa4x4x4, CommonMethods) {
  Mat3x8 xyz_global_i;
  xyz_global_i.row(0) << -1, +1, +1, -1, -1, +1, +1, -1;
  xyz_global_i.row(1) << -1, -1, +1, +1, -1, -1, +1, +1;
  xyz_global_i.row(2) << -1, -1, -1, -1, +1, +1, +1, +1;
  auto hexa = Hexa4x4x4(xyz_global_i);
  EXPECT_EQ(hexa.local_to_global_Dx1(1, 1, 1), Mat3x1(1, 1, 1));
  EXPECT_EQ(hexa.local_to_global_Dx1(1.5, 1.5, 1.5), Mat3x1(1.5, 1.5, 1.5));
  EXPECT_EQ(hexa.local_to_global_Dx1(3, 4, 5), Mat3x1(3, 4, 5));
  EXPECT_EQ(hexa.global_to_local_3x1(3, 4, 2), Mat3x1(3, 4, 2));
  EXPECT_EQ(hexa.global_to_local_3x1(4, 5.5, 2.5), Mat3x1(4, 5.5, 2.5));
  EXPECT_EQ(hexa.global_to_local_3x1(7, 13, 6), Mat3x1(7, 13, 6));
  EXPECT_DOUBLE_EQ(Quadrature([](Mat3x1 const&){ return 2.0; }, hexa), 16.0);
  EXPECT_DOUBLE_EQ(Integrate([](Mat3x1 const&){ return 2.0; }, hexa), 16.0);
  auto f = [](Mat3x1 const& xyz){ return xyz[0]; };
  auto g = [](Mat3x1 const& xyz){ return xyz[1]; };
  auto h = [](Mat3x1 const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, hexa), Integrate(h, hexa));
  EXPECT_DOUBLE_EQ(Norm(f, hexa), sqrt(Innerprod(f, f, hexa)));
  EXPECT_DOUBLE_EQ(Norm(g, hexa), sqrt(Innerprod(g, g, hexa)));
}
TEST_F(TestHexa4x4x4, Basis) {
  Mat3x1 origin = {0, 0, 0}, left = {-1, 2, 3}, right = {1, 3, 2};
  B b; double residual;
  b = B();
  EXPECT_EQ(b(origin), Y(1, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  EXPECT_EQ(b(left), Y(1, -1, 2, 3, 1, -2, -3, 4, 6, 9));
  EXPECT_EQ(b(right), Y(1, 1, 3, 2, 1, 3, 2, 9, 6, 4));
  Mat3x8 xyz_global_i;
  xyz_global_i.row(0) << -1, +1, +1, -1, -1, +1, +1, -1;
  xyz_global_i.row(1) << -1, -1, +1, +1, -1, -1, +1, +1;
  xyz_global_i.row(2) << -1, -1, -1, -1, +1, +1, +1, +1;
  auto hexa = Hexa4x4x4(xyz_global_i);
  Orthonormalize(&b, hexa);
  residual = (Integrate([&b](const Mat3x1& xyz) {
    auto col = b(xyz);
    A prod = col * col.transpose();
    return prod;
  }, hexa) - A::Identity()).cwiseAbs().maxCoeff();
  EXPECT_NEAR(residual, 0.0, 1e-14);
  b = B(left);
  EXPECT_EQ(b(origin), Y(1, 1, -2, -3, 1, -2, -3, 4, 6, 9));
  EXPECT_EQ(b(left), Y(1, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  EXPECT_EQ(b(right), Y(1, 2, 1, -1, 4, 2, -2, 1, -1, 1)); 
  auto x = left[0], y = left[1], z = left[2];
  xyz_global_i.row(0) << x-1, x+1, x+1, x-1, x-1, x+1, x+1, x-1;
  xyz_global_i.row(1) << y-1, y-1, y+1, y+1, y-1, y-1, y+1, y+1;
  xyz_global_i.row(2) << z-1, z-1, z-1, z-1, z+1, z+1, z+1, z+1;
  hexa = Hexa4x4x4(xyz_global_i);
  b.Orthonormalize(hexa);
  residual = (Integrate([&b](const Mat3x1& xyz) {
    auto col = b(xyz);
    A prod = col * col.transpose();
    return prod;
  }, hexa) - A::Identity()).cwiseAbs().maxCoeff();
  EXPECT_NEAR(residual, 0.0, 1e-14);
}
TEST_F(TestHexa4x4x4, ProjFunc) {
  Mat3x8 xyz_global_i;
  xyz_global_i.row(0) << -1, +1, +1, -1, -1, +1, +1, -1;
  xyz_global_i.row(1) << -1, -1, +1, +1, -1, -1, +1, +1;
  xyz_global_i.row(2) << -1, -1, -1, -1, +1, +1, +1, +1;
  auto hexa = Hexa4x4x4(xyz_global_i);
  B b;
  Orthonormalize(&b, hexa);
  auto fscalar = [](Mat3x1 const& xyz){
    return xyz[0] * xyz[1] + xyz[2];
  };
  auto scalar = Pscalar(fscalar, b, hexa);
  double residual = (scalar.GetCoef() - Mat1x10(0, 0, 0, 1, 0, 1, 0, 0, 0, 0))
      .cwiseAbs().maxCoeff();
  EXPECT_NEAR(residual, 0.0, 1e-15);
  auto fvector = [](Mat3x1 const& xyz) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    Mat11x1 func(0, 1,
                x, y, z,
                x * x, x * y, x * z, y * y, y * z, z * z);
    return func;
  };
  auto vec = Pvector(fvector, b, hexa);
  Mat11x10 exact_vector;
  exact_vector.row(0).setZero();
  exact_vector.bottomRows(10).setIdentity();
  auto abs_diff = (vec.GetCoef() - exact_vector).cwiseAbs();
  EXPECT_NEAR(abs_diff.maxCoeff(), 0.0, 1e-14);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

}  // namespace integrator
}  // namespace mini
