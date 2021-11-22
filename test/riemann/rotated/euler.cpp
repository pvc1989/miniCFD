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
  static void ExpectNear(double x, double y, double eps) {
    if (x == 0) {
      EXPECT_EQ(y, 0);
    } else {
      EXPECT_NEAR(y / x, 1, eps);
    }
  }
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
  using Solver = Euler<euler::Exact<Gas, 3>>;
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
TEST_F(TestRotatedEuler, Test3dSolver) {
  using Solver = Euler<euler::Exact<Gas, 3>>;
  using Scalar = Solver::Scalar;
  using State = Solver::State;
  using Speed = State::Speed;
  using Flux = Solver::Flux;

  auto CompareFlux = [](Flux const& lhs, Flux const& rhs) {
    constexpr double eps = 1e-4;
    ExpectNear(lhs.mass, rhs.mass, eps);
    ExpectNear(lhs.energy, rhs.energy, eps);
    ExpectNear(lhs.momentum[0], rhs.momentum[0], eps);
    ExpectNear(lhs.momentum[1], rhs.momentum[1], eps);
    ExpectNear(lhs.momentum[2], rhs.momentum[2], eps);
  };

  Solver solver;
  auto frame = algebra::Matrix<Scalar, 3, 3>();
  Speed v__left{1.5}, v_right{2.5};
  Speed w__left{1.5}, w_right{0.5};

  frame.col(0) << 1, 0, 0;
  frame.col(1) << 0, 1, 0;
  frame.col(2) << 0, 0, 1;
  solver.Rotate(frame);
  State  left{1.000, 0.0, v__left, w__left, 1.0};
  State right{0.125, 0.0, v_right, w_right, 0.1};
  auto left_c  = Gas::PrimitiveToConservative(left);
  auto right_c = Gas::PrimitiveToConservative(right);
  CompareFlux(solver.GetFluxOnTimeAxis(left_c, right_c),
      solver.GetFlux({0.426319, +0.927453, v__left, w__left, 0.303130}));
  CompareFlux(solver.GetFluxOnTimeAxis(right_c, left_c),
      solver.GetFlux({0.426319, -0.927453, v__left, w__left, 0.303130}));

  frame.col(0) << 1, 0, 0;
  frame.col(1) << 0, -1, 0;
  frame.col(2) << 0, 0, -1;
  solver.Rotate(frame);
  CompareFlux(solver.GetFluxOnTimeAxis(left_c, right_c),
      solver.GetFlux({0.426319, +0.927453, v__left, w__left, 0.303130}));
  CompareFlux(solver.GetFluxOnTimeAxis(right_c, left_c),
      solver.GetFlux({0.426319, -0.927453, v__left, w__left, 0.303130}));
}

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
