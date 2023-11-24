// Copyright 2019 PEI Weicheng and YANG Minghao

#include <vector>

#include "gtest/gtest.h"

#include "mini/riemann/euler/types.hpp"

namespace mini {
namespace riemann {
namespace euler {

class TestTypes : public ::testing::Test {
};
TEST_F(TestTypes, TestTuples) {
  auto rho{0.1}, u{+0.3}, v{-0.4}, p{0.5};
  auto primitive = Primitives<double, 2>{rho, u, v, p};
  EXPECT_DOUBLE_EQ(primitive[0], primitive.rho());
  EXPECT_DOUBLE_EQ(primitive.rho(), primitive.mass());
  EXPECT_DOUBLE_EQ(primitive[1], primitive.u());
  EXPECT_DOUBLE_EQ(primitive.u(), primitive.momentumX());
  EXPECT_DOUBLE_EQ(primitive[2], primitive.v());
  EXPECT_DOUBLE_EQ(primitive.v(), primitive.momentumY());
  EXPECT_DOUBLE_EQ(primitive[3], primitive.p());
  EXPECT_DOUBLE_EQ(primitive.p(), primitive.energy());
  using Vector = typename Primitives<double, 2>::Vector;
  EXPECT_EQ(primitive.momentum(), Vector(+0.3, -0.4));
  EXPECT_DOUBLE_EQ(primitive.GetDynamicPressure(), rho * (u*u + v*v) / 2);
}
TEST_F(TestTypes, TestConverters) {
  auto rho{0.1}, u{+0.2}, v{-0.2}, p{0.3};
  auto primitive = Primitives<double, 2>{rho, u, v, p};
  using Gas = IdealGas<double, 1.4>;
  constexpr auto gamma = Gas::Gamma();
  auto conservative = Conservatives<double, 2>{
    rho, rho*u, rho*v, p/(gamma-1) + 0.5*rho*(u*u + v*v)
  };
  EXPECT_EQ(Gas::PrimitiveToConservative(primitive), conservative);
  auto primitive_copy = Gas::ConservativeToPrimitive(conservative);
  EXPECT_DOUBLE_EQ(primitive_copy.rho(), rho);
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
