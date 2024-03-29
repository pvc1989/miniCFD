// Copyright 2019 PEI Weicheng and YANG Minghao

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
  using Gas = euler::IdealGas<double, 1.4>;
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
  using Vector = Solver::Vector;
  using Frame = Solver::Frame;
  using Value = Solver::Value;
  Solver solver;
  Frame frame { Vector{+0.6, 0.8}, Vector{-0.8, 0.6} };
  Value v{0, 3.0, 4.0, 0}, v_copy{0, 3.0, 4.0, 0};
  solver.Rotate(frame);
  solver.GlobalToNormal(&v);
  Vector momentum = v_copy.momentum();
  EXPECT_EQ(v.momentumX(), momentum.dot(frame[0]));
  EXPECT_NEAR(v.momentumY(), momentum.dot(frame[1]), 1e-15);
  solver.NormalToGlobal(&v);
  EXPECT_DOUBLE_EQ(v[0], v_copy[0]);
  EXPECT_DOUBLE_EQ(v[1], v_copy[1]);
  EXPECT_DOUBLE_EQ(v[2], v_copy[2]);
  EXPECT_DOUBLE_EQ(v[3], v_copy[3]);
}
TEST_F(TestRotatedEuler, Test3dConverter) {
  using Solver = Euler<euler::Exact<Gas, 3>>;
  using Scalar = Solver::Scalar;
  using Vector = Solver::Vector;
  using Frame = Solver::Frame;
  using Value = Solver::Value;
  Solver solver;
  Frame frame {
    Vector{+0.6, 0.8, 0.0}, Vector{-0.8, 0.6, 0.0}, Vector{0.0, 0.0, 1.0}
  };
  Value v{0, 3.0, 4.0, 5.0, 0}, v_copy{0, 3.0, 4.0, 5.0, 0};
  solver.Rotate(frame);
  solver.GlobalToNormal(&v);
  Vector momentum = v_copy.momentum();
  EXPECT_EQ(v.momentumX(), momentum.dot(frame[0]));
  EXPECT_EQ(v.momentumY(), momentum.dot(frame[1]));
  EXPECT_EQ(v.momentumZ(), momentum.dot(frame[2]));
  solver.NormalToGlobal(&v);
  EXPECT_DOUBLE_EQ(v[0], v_copy[0]);
  EXPECT_DOUBLE_EQ(v[1], v_copy[1]);
  EXPECT_DOUBLE_EQ(v[2], v_copy[2]);
  EXPECT_DOUBLE_EQ(v[3], v_copy[3]);
  EXPECT_DOUBLE_EQ(v[4], v_copy[4]);
}
TEST_F(TestRotatedEuler, Test3dSolver) {
  using Solver = Euler<euler::Exact<Gas, 3>>;
  using Vector = Solver::Vector;
  using Scalar = Solver::Scalar;
  using Primitive = Solver::Primitive;
  using Speed = Scalar;
  using Flux = Solver::Flux;

  auto CompareFlux = [](Flux const& lhs, Flux const& rhs) {
    constexpr double eps = 1e-4;
    ExpectNear(lhs.mass(), rhs.mass(), eps);
    ExpectNear(lhs.energy(), rhs.energy(), eps);
    ExpectNear(lhs.momentumX(), rhs.momentumX(), eps);
    ExpectNear(lhs.momentumY(), rhs.momentumY(), eps);
    ExpectNear(lhs.momentumZ(), rhs.momentumZ(), eps);
  };

  Solver solver;
  auto frame = std::array<Vector, 3>();
  Speed v__left{1.5}, v_right{2.5};
  Speed w__left{1.5}, w_right{0.5};

  frame[0] = { 1, 0, 0 };
  frame[1] = { 0, 1, 0 };
  frame[2] = { 0, 0, 1 };
  solver.Rotate(frame);
  Primitive  left{1.000, 0.0, v__left, w__left, 1.0};
  Primitive right{0.125, 0.0, v_right, w_right, 0.1};
  auto left_c  = Gas::PrimitiveToConservative(left);
  auto right_c = Gas::PrimitiveToConservative(right);
  CompareFlux(solver.GetFluxUpwind(left_c, right_c),
      solver.GetFlux({0.426319, +0.927453, v__left, w__left, 0.303130}));
  CompareFlux(solver.GetFluxUpwind(right_c, left_c),
      solver.GetFlux({0.426319, -0.927453, v__left, w__left, 0.303130}));

  frame[1] = { 0, -1, 0 };
  frame[2] = { 0, 0, -1 };
  solver.Rotate(frame);
  CompareFlux(solver.GetFluxUpwind(left_c, right_c),
      solver.GetFlux({0.426319, +0.927453, v__left, w__left, 0.303130}));
  CompareFlux(solver.GetFluxUpwind(right_c, left_c),
      solver.GetFlux({0.426319, -0.927453, v__left, w__left, 0.303130}));
}

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
