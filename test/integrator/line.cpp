#include "mini/integrator/line.hpp"

#include "gtest/gtest.h"

using Scalar = double;
using Mat3x1 = Eigen::Matrix<Scalar, 3, 1>;

constexpr int n_func(int d) {
  return d == 1 ? 3 : (d == 2 ? 6 : 10);
}

template <int D>
auto raw_basis(Eigen::Matrix<Scalar, D, 1> xyz);

template <>
auto raw_basis(Eigen::Matrix<Scalar, 1, 1> xyz) {
  Scalar x = xyz[0];
  Eigen::Matrix<Scalar, n_func(1), 1> basis = {
    1, x, x * x
  };
  return basis;
}

template <>
auto raw_basis(Eigen::Matrix<Scalar, 2, 1> xyz) {
  Scalar x = xyz[0], y = xyz[1];
  Eigen::Matrix<Scalar, n_func(2), 1> basis = {
    1, 
    x, y+1,
    x * x, x * y+x, y * y + 2 * y + 1
  };
  return basis;
}

template <>
auto raw_basis(Eigen::Matrix<Scalar, 3, 1> xyz) {
  Scalar x = xyz[0], y = xyz[1], z = xyz[2];
  Eigen::Matrix<Scalar, n_func(3), 1> basis = {
    1,
    x, y, z,
    x * x, x * y, x * z,
    y * y, y * z, z * z,
  };
  return basis;
}

template <int D>
void test() {
  using MatDx2 = Eigen::Matrix<Scalar, D, 2>;
  using MatDx1 = Eigen::Matrix<Scalar, D, 1>;
  MatDx2 xyz_global_i;
  for (int d = 0; d < D; ++d)
    xyz_global_i.row(d) << -2, 1;
  auto line = Line<Scalar, 4, D>(xyz_global_i);
  print(line.xyz_global_Dx2_);
  print(line.local_to_global_Dx1(-1));
  print(line.local_to_global_Dx1(0));
  print(line.local_to_global_Dx1(1));

  // print(line.global_to_local_Dx1(-2));
  // print(line.global_to_local_Dx1(-0.5));
  // print(line.global_to_local_Dx1(1));

  print(line.integrate([](MatDx1){ return 2.0; }));

  auto schmidt = line.template orthonormalize<n_func(D)>(raw_basis<D>);
  print("schmidt = ");
  print(schmidt);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
  // test<1>();
  test<2>();
  // test<3>();
}
