// Copyright 2019 Weicheng Pei and Minghao Yang

#include <vector>

#include "gtest/gtest.h"

#include "mini/riemann/euler/types.hpp"

namespace mini {
namespace riemann {
namespace euler {

class TestIdealGas : public ::testing::Test {
 protected:
  using Gas = IdealGas<double, 1, 4>;
  static constexpr auto gamma = Gas::Gamma();
};
TEST_F(TestIdealGas, TestConverters) {
  auto rho{0.1}, u{+0.2}, v{-0.2}, p{0.3};
  auto primitive = PrimitiveTuple<2, double>{rho, u, v, p};
  auto conservative = ConservativeTuple<2, double>{
    rho, rho*u, rho*v, p/(gamma-1) + 0.5*rho*(u*u + v*v)
  };
  EXPECT_EQ(Gas::PrimitiveToConservative(primitive), conservative);
  auto primitive_copy = Gas::ConservativeToPrimitive(conservative);
  EXPECT_EQ(primitive_copy.rho(), rho);
  EXPECT_DOUBLE_EQ(primitive_copy.u(), u);
  EXPECT_DOUBLE_EQ(primitive_copy.v(), v);
  EXPECT_DOUBLE_EQ(primitive_copy.p(), p);
}

}  // namespace euler
}  // namespace riemann
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
