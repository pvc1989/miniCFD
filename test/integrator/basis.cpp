//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <iostream>

#include "mini/integrator/basis.hpp"
#include "mini/integrator/function.hpp"
#include "mini/integrator/hexa.hpp"

#include "gtest/gtest.h"


namespace mini {
namespace integrator {

class TestProjFunc : public ::testing::Test {
 protected:
  using Hexa4x4x4 = Hexa<double, 4, 4, 4>;
  using Mat3x1 = algebra::Matrix<double, 3, 1>;
  using Mat3x8 = algebra::Matrix<double, 3, 8>;
  using BasisType = Basis<double, 3, 2>;
  static constexpr int N = BasisType::N;
  using MatNx1 = algebra::Matrix<double, N, 1>;
  static constexpr int K = 10;
  using MatKxN = algebra::Matrix<double, K, N>;
  static MatKxN GetMpdv(double x, double y, double z) {
    MatKxN mat_pdv;
    for (int i = 1; i < N; ++i)
      mat_pdv(i, i) = 1;
    mat_pdv(4, 1) = 2 * x;
    mat_pdv(5, 1) = y;
    mat_pdv(6, 1) = z;
    mat_pdv(5, 2) = x;
    mat_pdv(7, 2) = 2 * y;
    mat_pdv(8, 2) = z;
    mat_pdv(6, 3) = x;
    mat_pdv(8, 3) = y;
    mat_pdv(9, 3) = 2 * z;
    return mat_pdv;
  }
};
TEST_F(TestProjFunc, Derivative) {
  Mat3x1 origin = {0, 0, 0};
  Mat3x8 xyz_global_i;
  xyz_global_i.row(0) << -1, +1, +1, -1, -1, +1, +1, -1;
  xyz_global_i.row(1) << -1, -1, +1, +1, -1, -1, +1, +1;
  xyz_global_i.row(2) << -1, -1, -1, -1, +1, +1, +1, +1;
  auto hexa = Hexa4x4x4(xyz_global_i);
  auto basis = BasisType();
  Orthonormalize(&basis, hexa);
  auto fvector = [](Mat3x1 const& xyz) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    MatNx1 func(1, x, y, z, x * x, x * y, x * z, y * y, y * z, z * z);
    return func;
  };
  using Pvector = ProjFunc<double, 3, 2, 10>;
  auto pf_vec = Pvector(fvector, basis, hexa);
  double x = 0.0, y = 0.0, z = 0.0;
  auto mat_actual = pf_vec.GetMpdv({x, y, z});
  std::cout << "mat_actual =\n" << mat_actual << std::endl;
  auto mat_expect = GetMpdv(x, y, z);
  std::cout << "mat_expect =\n" << mat_expect << std::endl;
  MatKxN diff = mat_actual - mat_expect;
  EXPECT_NEAR(diff.cwiseAbs().maxCoeff(), 0.0, 1e-14);
  auto s_actual = pf_vec.GetSmoothness(hexa);
  std::cout << "s_actual =\n" << s_actual << std::endl;
  double eps = 1e-12;
  EXPECT_NEAR(s_actual(0), 0.0, eps);
  EXPECT_NEAR(s_actual(1), 8.0, eps);
  EXPECT_NEAR(s_actual(2), 8.0, eps);
  EXPECT_NEAR(s_actual(3), 8.0, eps);
  EXPECT_NEAR(s_actual(4), 80.0/3, eps);
  EXPECT_NEAR(s_actual(5), 64.0/3, eps);
  EXPECT_NEAR(s_actual(6), 64.0/3, eps);
  EXPECT_NEAR(s_actual(7), 80.0/3, eps);
  EXPECT_NEAR(s_actual(8), 64.0/3, eps);
  EXPECT_NEAR(s_actual(9), 80.0/3, eps);
}
int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

}  // namespace integrator
}  // namespace mini
