// Copyright 2019 PEI Weicheng and YANG Minghao

#include <vector>

#include "gtest/gtest.h"

#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/euler/exact.hpp"

namespace mini {
namespace riemann {
namespace euler {

class TestExact : public ::testing::Test {
 protected:
  using Gas = IdealGas<double, 1.4>;
  using Solver = Exact<Gas, 1>;
  using Primitive = Solver::Primitive;
  using Flux = Solver::Flux;
  Solver solver;
  static void ExpectNear(double x, double y, double eps) {
    if (x == 0) {
      EXPECT_EQ(y, 0);
    } else {
      EXPECT_NEAR(y / x, 1, eps);
    }
  }
  static void CompareFlux(Flux const& lhs, Flux const& rhs) {
    constexpr double eps = 1e-4;
    ExpectNear(lhs.mass(), rhs.mass(), eps);
    ExpectNear(lhs.energy(), rhs.energy(), eps);
    ExpectNear(lhs.momentumX(), rhs.momentumX(), eps);
  }
};
TEST_F(TestExact, TestFlux) {
  auto rho{0.1}, u{0.2}, p{0.3};
  auto flux = Flux{rho * u, rho * u * u + p, u};
  flux.energy() *= p * Gas::GammaOverGammaMinusOne() + 0.5 * rho * u * u;
  EXPECT_EQ(solver.GetFlux({rho, u, p}), flux);
}
TEST_F(TestExact, TestEqualStates) {
  Primitive left{1.0, 0.0, 1.0};
  Primitive const& right = left;
  CompareFlux(solver.GetFlux(left), solver.GetFlux(right));
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux(left));
  left.u() = +1.0;
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux(left));
  left.u() = -1.0;
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux(left));
}
TEST_F(TestExact, TestSod) {
  Primitive left{1.0, 0.0, 1.0}, right{0.125, 0.0, 0.1};
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({0.426319, +0.927453, 0.303130}));
  CompareFlux(solver.GetFluxUpwind(right, left),
              solver.GetFlux({0.426319, -0.927453, 0.303130}));
}
TEST_F(TestExact, TestShockCollision) {
  Primitive left{5.99924, 19.5975, 460.894}, right{5.99242, 6.19633, 46.0950};
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({5.99924, 19.5975, 460.894}));
}
TEST_F(TestExact, TestBlastFromLeft) {
  Primitive left{1.0, 0.0, 1000}, right{1.0, 0.0, 0.01};
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({0.575062, 19.59745, 460.8938}));
}
TEST_F(TestExact, TestBlastFromRight) {
  Primitive left{1.0, 0.0, 0.01}, right{1.0, 0.0, 100};
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({0.575113, -6.196328, 46.09504}));
}
TEST_F(TestExact, TestAlmostVacuumed) {
  Primitive left{1.0, -2.0, 0.4}, right{1.0, +2.0, 0.4};
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({0.21852, 0.0, 0.001894}));
}
TEST_F(TestExact, TestVacuumed) {
  Primitive left{1.0, -4.0, 0.4}, right{1.0, +4.0, 0.4};
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({0.0, 0.0, 0.0}));
}
TEST_F(TestExact, TestRightVacuum) {
  // u_* = u_L + \frac{2 a_L}{1.4 - 1} = u_L + 5 a_L
  Primitive left{1.4, 100.0, 1.0}, right{0, 100.0, 0};
  EXPECT_DOUBLE_EQ(Gas::GetSpeedOfSound(left), 1.0);
  Flux flux_actual, flux_expect;
  // Wave[1] <<< Axis[t]
  left.u() = -6;  // [u_L - a_L, u_L + 5 a_L] = [-7, -1]
  flux_actual = solver.GetFluxUpwind(left, right);
  flux_expect = solver.GetFlux(right);
  EXPECT_DOUBLE_EQ(flux_actual.mass(), flux_expect.mass());
  EXPECT_DOUBLE_EQ(flux_actual.momentumX(), flux_expect.momentumX());
  EXPECT_DOUBLE_EQ(flux_actual.energy(), flux_expect.energy());
  left.u() = -5;  // [u_L - a_L, u_L + 5 a_L] = [-6, 0]
  flux_actual = solver.GetFluxUpwind(left, right);
  flux_expect = solver.GetFlux(right);
  EXPECT_NEAR(flux_actual.mass(), flux_expect.mass(), 1e-18);
  EXPECT_NEAR(flux_actual.momentumX(), flux_expect.momentumX(), 1e-18);
  EXPECT_NEAR(flux_actual.energy(), flux_expect.energy(), 1e-18);
  // Axis[t] inside  Wave[1]
  left.u() = -4;  // [u_L - a_L, u_L + 5 a_L] = [-5, 1]
  flux_actual = solver.GetFluxUpwind(left, right);
  auto rho = 1.4 * std::pow(6.0, -5);
  auto u = std::pow(6.0, -1);
  auto p = std::pow(6.0, -7);
  flux_expect = solver.GetFlux(Primitive(rho, u, p));
  EXPECT_NEAR(flux_actual.mass(), flux_expect.mass(), 1e-19);
  EXPECT_NEAR(flux_actual.momentumX(), flux_expect.momentumX(), 1e-19);
  EXPECT_NEAR(flux_actual.energy(), flux_expect.energy(), 1e-20);
  // Axis[t] <<< Wave[1]
  left.u() = 1;  // [u_L - a_L, u_L + 5 a_L] = [0, 6]
  flux_actual = solver.GetFluxUpwind(left, right);
  flux_expect = solver.GetFlux(left);
  EXPECT_NEAR(flux_actual.mass(), flux_expect.mass(), 1e-14);
  EXPECT_NEAR(flux_actual.momentumX(), flux_expect.momentumX(), 1e-14);
  EXPECT_NEAR(flux_actual.energy(), flux_expect.energy(), 1e-14);
  left.u() = 2;  // [u_L - a_L, u_L + 5 a_L] = [1, 7]
  flux_actual = solver.GetFluxUpwind(left, right);
  flux_expect = solver.GetFlux(left);
  EXPECT_NEAR(flux_actual.mass(), flux_expect.mass(), 1e-14);
  EXPECT_NEAR(flux_actual.momentumX(), flux_expect.momentumX(), 1e-14);
  EXPECT_NEAR(flux_actual.energy(), flux_expect.energy(), 1e-14);
}

class TestExact2d : public ::testing::Test {
 protected:
  using Solver = Exact<IdealGas<double, 1.4>, 2>;
  using Primitive = Solver::Primitive;
  using Speed = Solver::Scalar;
  using Flux = Solver::Flux;
  Solver solver;
  Speed v__left{1.5}, v_right{2.5};
  static void ExpectNear(double x, double y, double eps) {
    if (x == 0) {
      EXPECT_EQ(y, 0);
    } else {
      EXPECT_NEAR(y / x, 1, eps);
    }
  }
  static void CompareFlux(Flux const& lhs, Flux const& rhs) {
    constexpr double eps = 1e-4;
    ExpectNear(lhs.mass(), rhs.mass(), eps);
    ExpectNear(lhs.energy(), rhs.energy(), eps);
    ExpectNear(lhs.momentumX(), rhs.momentumX(), eps);
    ExpectNear(lhs.momentumY(), rhs.momentumY(), eps);
  }
};
TEST_F(TestExact2d, TestEqualStates) {
  Primitive  left{1.0, 0.0, v__left, 1.0};
  Primitive right{1.0, 0.0, v_right, 1.0};
  CompareFlux(solver.GetFlux(left), solver.GetFlux(right));
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux(left));
  left.u() = +1.0;
  right.u() = left.u();
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux(left));
  left.u() = -1.0;
  right.u() = left.u();
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux(right));
}
TEST_F(TestExact2d, TestSod) {
  Primitive  left{1.000, 0.0, v__left, 1.0};
  Primitive right{0.125, 0.0, v_right, 0.1};
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({0.426319, +0.927453, v__left, 0.303130}));
  CompareFlux(solver.GetFluxUpwind(right, left),
              solver.GetFlux({0.426319, -0.927453, v__left, 0.303130}));
}
TEST_F(TestExact2d, TestShockCollision) {
  Primitive  left{5.99924, 19.59750, v__left, 460.894};
  Primitive right{5.99242, -6.19633, v_right, 46.0950};
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({5.99924, 19.5975, v__left, 460.894}));
}
TEST_F(TestExact2d, TestBlastFromLeft) {
  Primitive  left{1.0, 0.0, v__left, 1e+3};
  Primitive right{1.0, 0.0, v_right, 1e-2};
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({0.575062, 19.59745, v__left, 460.8938}));
}
TEST_F(TestExact2d, TestBlastFromRight) {
  Primitive  left{1.0, 0.0, v__left, 1e-2};
  Primitive right{1.0, 0.0, v_right, 1e+2};
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({0.575113, -6.196328, v_right, 46.09504}));
}
TEST_F(TestExact2d, TestAlmostVacuumed) {
  Primitive  left{1.0, -2.0, v__left, 0.4};
  Primitive right{1.0, +2.0, v_right, 0.4};
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({0.21852, 0.0, v__left, 0.001894}));
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({0.21852, 0.0, v_right, 0.001894}));
}
TEST_F(TestExact2d, TestVacuumed) {
  Primitive  left{1.0, -4.0, v__left, 0.4};
  Primitive right{1.0, +4.0, v_right, 0.4};
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({0.0, 0.0, v__left, 0.0}));
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({0.0, 0.0, v_right, 0.0}));
}

class TestExact3d : public ::testing::Test {
 protected:
  using Gas = IdealGas<double, 1.4>;
  using Solver = Exact<Gas, 3>;
  using Primitive = Solver::Primitive;
  using Speed = Solver::Scalar;
  using Flux = Solver::Flux;
  Solver solver;
  Speed v__left{1.5}, v_right{2.5};
  Speed w__left{1.5}, w_right{0.5};
  static void ExpectNear(double x, double y, double eps) {
    if (x == 0) {
      EXPECT_EQ(y, 0);
    } else {
      EXPECT_NEAR(y / x, 1, eps);
    }
  }
  static void CompareFlux(Flux const& lhs, Flux const& rhs) {
    constexpr double eps = 1e-4;
    ExpectNear(lhs.mass(), rhs.mass(), eps);
    ExpectNear(lhs.energy(), rhs.energy(), eps);
    ExpectNear(lhs.momentumX(), rhs.momentumX(), eps);
    ExpectNear(lhs.momentumY(), rhs.momentumY(), eps);
    ExpectNear(lhs.momentumZ(), rhs.momentumZ(), eps);
  }
};
TEST_F(TestExact3d, TestEqualStates) {
  Primitive  left{1.0, 0.0, v__left, w__left, 1.0};
  Primitive right{1.0, 0.0, v_right, w_right, 1.0};
  CompareFlux(solver.GetFlux(left), solver.GetFlux(right));
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux(left));
  left.u() = +1.0;
  right.u() = left.u();
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux(left));
  left.u() = -1.0;
  right.u() = left.u();
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux(right));
}
TEST_F(TestExact3d, TestSod) {
  Primitive  left{1.000, 0.0, v__left, w__left, 1.0};
  Primitive right{0.125, 0.0, v_right, w_right, 0.1};
  CompareFlux(solver.GetFluxUpwind(left, right),
      solver.GetFlux({0.426319, +0.927453, v__left, w__left, 0.303130}));
  CompareFlux(solver.GetFluxUpwind(right, left),
      solver.GetFlux({0.426319, -0.927453, v__left, w__left, 0.303130}));
}
TEST_F(TestExact3d, TestShockCollision) {
  Primitive  left{5.99924, 19.59750, v__left, w__left, 460.894};
  Primitive right{5.99242, -6.19633, v_right, w_right, 46.0950};
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({5.99924, 19.5975, v__left, w__left, 460.894}));
}
TEST_F(TestExact3d, TestBlastFromLeft) {
  Primitive  left{1.0, 0.0, v__left, w__left, 1e+3};
  Primitive right{1.0, 0.0, v_right, w_right, 1e-2};
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({0.575062, 19.59745, v__left, w__left, 460.8938}));
}
TEST_F(TestExact3d, TestBlastFromRight) {
  Primitive  left{1.0, 0.0, v__left, w__left, 1e-2};
  Primitive right{1.0, 0.0, v_right, w_right, 1e+2};
  CompareFlux(solver.GetFluxUpwind(left, right),
      solver.GetFlux({0.575113, -6.196328, v_right, w_right, 46.09504}));
}
TEST_F(TestExact3d, TestAlmostVacuumed) {
  Primitive  left{1.0, -2.0, v__left, w__left, 0.4};
  Primitive right{1.0, +2.0, v_right, w_right, 0.4};
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({0.21852, 0.0, v__left, w__left, 0.001894}));
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({0.21852, 0.0, v_right, w_right, 0.001894}));
}
TEST_F(TestExact3d, TestVacuumed) {
  Primitive  left{1.0, -4.0, v__left, w__left, 0.4};
  Primitive right{1.0, +4.0, v_right, w_right, 0.4};
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({0.0, 0.0, v__left, w__left, 0.0}));
  CompareFlux(solver.GetFluxUpwind(left, right),
              solver.GetFlux({0.0, 0.0, v_right, w_right, 0.0}));
}
TEST_F(TestExact3d, TestLeftVacuum) {
  // u_* = u_R - \frac{2 a_R}{1.4 - 1} = u_R - 5 a_R
  double u{100.0}, v{50.0}, w{20};
  Primitive left{0, -u, -v, -w, 0}, right{1.4, u, v, w, 1.0};
  EXPECT_DOUBLE_EQ(Gas::GetSpeedOfSound(right), 1.0);
  Flux flux_actual, flux_expect;
  // Axis[t] <<< Wave[3]
  right.u() = +6;  // [u_R - 5 a_R, u_R + a_R] = [+1, +7]
  flux_actual = solver.GetFluxUpwind(left, right);
  flux_expect = solver.GetFlux(left);
  EXPECT_DOUBLE_EQ(flux_actual.mass(), flux_expect.mass());
  EXPECT_DOUBLE_EQ(flux_actual.momentumX(), flux_expect.momentumX());
  EXPECT_DOUBLE_EQ(flux_actual.momentumY(), flux_expect.momentumY());
  EXPECT_DOUBLE_EQ(flux_actual.momentumZ(), flux_expect.momentumZ());
  EXPECT_DOUBLE_EQ(flux_actual.energy(), flux_expect.energy());
  right.u() = +5;  // [u_R - 5 a_R, u_R + a_R] = [+0, +6]
  flux_actual = solver.GetFluxUpwind(left, right);
  flux_expect = solver.GetFlux(left);
  EXPECT_NEAR(flux_actual.mass(), flux_expect.mass(), 1e-18);
  EXPECT_NEAR(flux_actual.momentumX(), flux_expect.momentumX(), 1e-18);
  EXPECT_NEAR(flux_actual.momentumY(), flux_expect.momentumY(), 1e-18);
  EXPECT_NEAR(flux_actual.momentumZ(), flux_expect.momentumZ(), 1e-18);
  EXPECT_NEAR(flux_actual.energy(), flux_expect.energy(), 1e-18);
  // Axis[t] inside  Wave[3]
  right.u() = +4;  // [u_R - 5 a_R, u_R + a_R] = [-1, +5]
  flux_actual = solver.GetFluxUpwind(left, right);
  Primitive primitive;
  primitive.rho() = 1.4 * std::pow(6.0, -5);
  primitive.u() = -std::pow(6.0, -1); primitive.v() = v; primitive.w() = w;
  primitive.p() = std::pow(6.0, -7);
  flux_expect = solver.GetFlux(primitive);
  EXPECT_NEAR(flux_actual.mass(), flux_expect.mass(), 1e-19);
  EXPECT_NEAR(flux_actual.momentumX(), flux_expect.momentumX(), 1e-19);
  EXPECT_NEAR(flux_actual.momentumY(), flux_expect.momentumY(), 1e-17);
  EXPECT_NEAR(flux_actual.momentumZ(), flux_expect.momentumZ(), 1e-17);
  EXPECT_NEAR(flux_actual.energy(), flux_expect.energy(), 1e-16);
  // Wave[3] <<< Axis[t]
  right.u() = -1;  // [u_R - 5 a_R, u_R + a_R] = [-6, 0]
  flux_actual = solver.GetFluxUpwind(left, right);
  flux_expect = solver.GetFlux(right);
  EXPECT_NEAR(flux_actual.mass(), flux_expect.mass(), 1e-14);
  EXPECT_NEAR(flux_actual.momentumX(), flux_expect.momentumX(), 1e-14);
  EXPECT_NEAR(flux_actual.momentumY(), flux_expect.momentumY(), 1e-14);
  EXPECT_NEAR(flux_actual.momentumZ(), flux_expect.momentumZ(), 1e-14);
  EXPECT_NEAR(flux_actual.energy(), flux_expect.energy(), 1e-12);
  right.u() = -2;  // [u_R - 5 a_R, u_R + a_R] = [-7, -1]
  flux_actual = solver.GetFluxUpwind(left, right);
  flux_expect = solver.GetFlux(right);
  EXPECT_NEAR(flux_actual.mass(), flux_expect.mass(), 1e-14);
  EXPECT_NEAR(flux_actual.momentumX(), flux_expect.momentumX(), 1e-14);
  EXPECT_NEAR(flux_actual.momentumY(), flux_expect.momentumY(), 1e-18);
  EXPECT_NEAR(flux_actual.momentumZ(), flux_expect.momentumZ(), 1e-18);
  EXPECT_NEAR(flux_actual.energy(), flux_expect.energy(), 1e-14);
}

}  // namespace euler
}  // namespace riemann
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
