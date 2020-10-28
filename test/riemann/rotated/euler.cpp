// Copyright 2019 Weicheng Pei and Minghao Yang

#include <vector>

#include "gtest/gtest.h"

#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/euler/exact.hpp"
#include "mini/riemann/rotated/euler.hpp"

namespace mini {
namespace riemann {
namespace rotated {

class TestRotatedEulerTest : public ::testing::Test {
 protected:
  using Gas = euler::IdealGas<1, 4>;
  using UnrotatedSolver = euler::Exact<Gas, 2>;
  using Solver = Euler<UnrotatedSolver>;
  using Scalar = Solver::Scalar;
  using Vector = Solver::Vector;
  using State = Solver::State;
  Solver solver;
};
TEST_F(TestRotatedEulerTest, TestVectorConverter) {
  Vector n{+0.6, 0.8}, t{-0.8, 0.6}, v{3.0, 4.0}, v_copy{3.0, 4.0};
  solver.Rotate(n);
  solver.GlobalToNormal(&v);
  EXPECT_EQ(v[0], v_copy.Dot(n));
  EXPECT_EQ(v[1], v_copy.Dot(t));
  solver.NormalToGlobal(&v);
  EXPECT_DOUBLE_EQ(v[0], v_copy[0]);
  EXPECT_DOUBLE_EQ(v[1], v_copy[1]);
}

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
