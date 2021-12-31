// Copyright 2019 Weicheng Pei and Minghao Yang

#include <vector>

#include "gtest/gtest.h"

#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/euler/hllc.hpp"

namespace mini {
namespace riemann {
namespace euler {

class TestHllc : public ::testing::Test {
 protected:
  using Gas = IdealGas<1, 4>;
  using Solver = Hllc<Gas>;
  using State = Solver::State;
  using Flux = Solver::Flux;
  Solver solver;
  static void CompareFlux(Flux const& lhs, Flux const& rhs) {
    EXPECT_DOUBLE_EQ(lhs.mass(), rhs.mass());
    EXPECT_DOUBLE_EQ(lhs.energy(), rhs.energy());
    EXPECT_DOUBLE_EQ(lhs.momentumX(), rhs.momentumX());
  }
};
TEST_F(TestHllc, TestFlux) {
  auto rho{0.1}, u{0.2}, p{0.3};
  auto flux = Flux{rho * u, rho * u * u + p, u};
  flux.energy() *= p * Gas::GammaOverGammaMinusOne() + 0.5 * rho * u * u;
  EXPECT_EQ(solver.GetFlux({rho, u, p}), flux);
}
TEST_F(TestHllc, TestSod) {
  State left{1.0, 0.0, 1.0}, right{0.125, 0.0, 0.1};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.426319, +0.927453, 0.303130}));
  CompareFlux(solver.GetFluxOnTimeAxis(right, left),
              solver.GetFlux({0.426319, -0.927453, 0.303130}));
}
TEST_F(TestHllc, TestShockCollision) {
  State left{5.99924, 19.5975, 460.894}, right{5.99242, 6.19633, 46.0950};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({5.99924, 19.5975, 460.894}));
}
TEST_F(TestHllc, TestBlastFromLeft) {
  State left{1.0, 0.0, 1000}, right{1.0, 0.0, 0.01};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.575062, 19.59745, 460.8938}));
}
TEST_F(TestHllc, TestBlastFromRight) {
  State left{1.0, 0.0, 0.01}, right{1.0, 0.0, 100};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.575113, -6.196328, 46.09504}));
}
TEST_F(TestHllc, TestAlmostVaccumed) {
  State left{1.0, -2.0, 0.4}, right{1.0, +2.0, 0.4};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.21852, 0.0, 0.001894}));
}
TEST_F(TestHllc, TestVaccumed) {
  State left{1.0, -4.0, 0.4}, right{1.0, +4.0, 0.4};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.0, 0.0, 0.0}));
}

class TestHllc2d : public ::testing::Test {
 protected:
  using Solver = Hllc<IdealGas<1, 4>, 2>;
  using State = Solver::State;
  using Speed = State::Speed;
  using Flux = Solver::Flux;
  Solver solver;
  Speed v__left{1.5}, v_right{2.5};
  static void CompareFlux(Flux const& lhs, Flux const& rhs) {
    EXPECT_DOUBLE_EQ(lhs.mass(), rhs.mass());
    EXPECT_DOUBLE_EQ(lhs.energy(), rhs.energy());
    EXPECT_DOUBLE_EQ(lhs.momentumX(), rhs.momentumX());
    EXPECT_DOUBLE_EQ(lhs.momentumY(), rhs.momentumY());
  }
};
TEST_F(TestHllc2d, TestSod) {
  State  left{1.000, 0.0, v__left, 1.0};
  State right{0.125, 0.0, v_right, 0.1};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.426319, +0.927453, v__left, 0.303130}));
  CompareFlux(solver.GetFluxOnTimeAxis(right, left),
              solver.GetFlux({0.426319, -0.927453, v__left, 0.303130}));
}
TEST_F(TestHllc2d, TestShockCollision) {
  State  left{5.99924, 19.5975, v__left, 460.894};
  State right{5.99242, 6.19633, v_right, 46.0950};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({5.99924, 19.5975, v__left, 460.894}));
}
TEST_F(TestHllc2d, TestBlastFromLeft) {
  State  left{1.0, 0.0, v__left, 1e+3};
  State right{1.0, 0.0, v_right, 1e-2};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.575062, 19.59745, v__left, 460.8938}));
}
TEST_F(TestHllc2d, TestBlastFromRight) {
  State  left{1.0, 0.0, v__left, 1e-2};
  State right{1.0, 0.0, v_right, 1e+2};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.575113, -6.196328, v_right, 46.09504}));
}
TEST_F(TestHllc2d, TestAlmostVaccumed) {
  State  left{1.0, -2.0, v__left, 0.4};
  State right{1.0, +2.0, v_right, 0.4};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.21852, 0.0, v__left, 0.001894}));
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.21852, 0.0, v_right, 0.001894}));
}
TEST_F(TestHllc2d, TestVaccumed) {
  State  left{1.0, -4.0, v__left, 0.4};
  State right{1.0, +4.0, v_right, 0.4};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.0, 0.0, v__left, 0.0}));
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.0, 0.0, v_right, 0.0}));
}

}  // namespace euler
}  // namespace riemann
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
