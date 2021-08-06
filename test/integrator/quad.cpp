//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include "mini/integrator/base.hpp"
#include "mini/integrator/quad.hpp"

#include "gtest/gtest.h"

using Scalar = double;

template <int D>
void test() {
  using MatDx4 = Eigen::Matrix<Scalar, D, 4>;
  using MatDx1 = Eigen::Matrix<Scalar, D, 1>;

  MatDx4 xyz_global_i;
  xyz_global_i.row(0) << -1, 1, 1, -1;
  xyz_global_i.row(1) << -1, -1, 1, 1;
  if (D == 3)
    xyz_global_i.row(2) << -1, -1, 1, 1;
  auto quad = Quad<Scalar, 4, D>(xyz_global_i);
  print(quad.local_to_global_Dx1(0, 0));
  print(quad.local_to_global_Dx1(1, 1));
  print(quad.local_to_global_Dx1(-1, -1));

  print(quad.integrate([](MatDx1){ return 2.0; }));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
  test<2>();
  test<3>();
}
