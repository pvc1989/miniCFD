// Copyright 2019 Weicheng Pei and Minghao Yang

#include <vector>

#include "gtest/gtest.h"

#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/euler/exact.hpp"
#include "mini/riemann/rotated/euler.hpp"

namespace mini {
namespace riemann {
namespace rotated {

class TestRotatedEuler : public ::testing::Test {
 protected:
  using Gas = euler::IdealGas<1, 4>;
};
TEST_F(TestRotatedEuler, Test2dConverter) {
  using UnrotatedSolver = euler::Exact<Gas, 2>;
  using Solver = Euler<UnrotatedSolver>;
  using Scalar = Solver::Scalar;
  using Vector = Solver::Vector;
  using State = Solver::State;
  Solver solver;
  Vector n{+0.6, 0.8}, t{-0.8, 0.6}, v{3.0, 4.0}, v_copy{3.0, 4.0};
  solver.Rotate(n);
  solver.GlobalToNormal(&v);
  EXPECT_EQ(v[0], v_copy.Dot(n));
  EXPECT_EQ(v[1], v_copy.Dot(t));
  solver.NormalToGlobal(&v);
  EXPECT_DOUBLE_EQ(v[0], v_copy[0]);
  EXPECT_DOUBLE_EQ(v[1], v_copy[1]);
}
TEST_F(TestRotatedEuler, Test3dConverter) {
  using UnrotatedSolver = euler::Exact<Gas, 3>;
  using Solver = Euler<UnrotatedSolver, 3>;
  using Scalar = Solver::Scalar;
  using Vector = Solver::Vector;
  using State = Solver::State;
  Solver solver;
  Vector nu{+0.6, 0.8, 0.0}, sigma{-0.8, 0.6, 0.0}, pi{0.0, 0.0, 1.0};
  Vector v{3.0, 4.0, 5.0}, v_copy{3.0, 4.0, 5.0};
  solver.Rotate(nu, sigma, pi);
  solver.GlobalToNormal(&v);
  EXPECT_EQ(v[0], v_copy.Dot(nu));
  EXPECT_EQ(v[1], v_copy.Dot(sigma));
  EXPECT_EQ(v[2], v_copy.Dot(pi));
  solver.NormalToGlobal(&v);
  EXPECT_DOUBLE_EQ(v[0], v_copy[0]);
  EXPECT_DOUBLE_EQ(v[1], v_copy[1]);
  EXPECT_DOUBLE_EQ(v[2], v_copy[2]);
}

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
