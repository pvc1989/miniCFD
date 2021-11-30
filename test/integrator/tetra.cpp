//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cmath>

#include "mini/integrator/function.hpp"
#include "mini/integrator/tetra.hpp"

#include "gtest/gtest.h"

using std::sqrt;

namespace mini {
namespace integrator {

class TestTetra46 : public ::testing::Test {
 protected:
  using Tetra46 = Tetra<double, 46>;
  using Mat1x4 = algebra::Matrix<double, 1, 4>;
  using Mat3x4 = algebra::Matrix<double, 3, 4>;
  using Mat3x1 = algebra::Matrix<double, 3, 1>;
};
TEST_F(TestTetra46, VirtualMethods) {
  Mat3x4 xyz_global_i;
  xyz_global_i.row(0) << 0, 3, 0, 0;
  xyz_global_i.row(1) << 0, 0, 3, 0;
  xyz_global_i.row(2) << 0, 0, 0, 3;
  auto tetra = Tetra46(xyz_global_i);
  static_assert(tetra.CellDim() == 3);
  static_assert(tetra.PhysDim() == 3);
  EXPECT_NEAR(tetra.volume(), 4.5, 1e-14);
  EXPECT_EQ(tetra.CountQuadPoints(), 46);
}
TEST_F(TestTetra46, CommonMethods) {
  Mat3x4 xyz_global_i;
  xyz_global_i.row(0) << 0, 3, 0, 0;
  xyz_global_i.row(1) << 0, 0, 3, 0;
  xyz_global_i.row(2) << 0, 0, 0, 3;
  auto tetra = Tetra46(xyz_global_i);
  EXPECT_EQ(tetra.LocalToGlobal(1, 0, 0), Mat3x1(0, 0, 0));
  EXPECT_EQ(tetra.LocalToGlobal(0, 1, 0), Mat3x1(3, 0, 0));
  EXPECT_EQ(tetra.LocalToGlobal(0, 0, 1), Mat3x1(0, 3, 0));
  EXPECT_EQ(tetra.LocalToGlobal(0, 0, 0), Mat3x1(0, 0, 3));
  EXPECT_EQ(tetra.global_to_local_3x1(0, 0, 0), Mat3x1(1, 0, 0));
  EXPECT_EQ(tetra.global_to_local_3x1(3, 0, 0), Mat3x1(0, 1, 0));
  EXPECT_EQ(tetra.global_to_local_3x1(0, 3, 0), Mat3x1(0, 0, 1));
  EXPECT_EQ(tetra.global_to_local_3x1(0, 0, 3), Mat3x1(0, 0, 0));
  EXPECT_NEAR(Quadrature([](Mat3x1 const&){ return 6.0; }, tetra), 1.0, 1e-14);
  EXPECT_NEAR(Integrate([](Mat3x1 const&){ return 6.0; }, tetra), 27.0, 1e-13);
  auto f = [](Mat3x1 const& xyz){ return xyz[0]; };
  auto g = [](Mat3x1 const& xyz){ return xyz[1]; };
  auto h = [](Mat3x1 const& xyz){ return xyz[0] * xyz[1]; };
  EXPECT_DOUBLE_EQ(Innerprod(f, g, tetra), Integrate(h, tetra));
  EXPECT_DOUBLE_EQ(Norm(f, tetra), sqrt(Innerprod(f, f, tetra)));
  EXPECT_DOUBLE_EQ(Norm(g, tetra), sqrt(Innerprod(g, g, tetra)));
}

}  // namespace integrator
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}