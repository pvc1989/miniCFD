// Copyright 2019 Weicheng Pei and Minghao Yang

#include <vector>

#include "gtest/gtest.h"

#include "mini/riemann/euler/types.hpp"
#include "mini/riemann/euler/exact.hpp"

namespace mini {
namespace riemann {
namespace euler {

class TestExact : public ::testing::Test {
 protected:
  using Gas = IdealGas<1, 4>;
  using Solver = Exact<Gas, 1>;
  using State = Solver::State;
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
  State left{1.0, 0.0, 1.0};
  State const& right = left;
  CompareFlux(solver.GetFlux(left), solver.GetFlux(right));
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux(left));
  left.u() = +1.0;
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux(left));
  left.u() = -1.0;
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux(left));
}
TEST_F(TestExact, TestSod) {
  State left{1.0, 0.0, 1.0}, right{0.125, 0.0, 0.1};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.426319, +0.927453, 0.303130}));
  CompareFlux(solver.GetFluxOnTimeAxis(right, left),
              solver.GetFlux({0.426319, -0.927453, 0.303130}));
}
TEST_F(TestExact, TestShockCollision) {
  State left{5.99924, 19.5975, 460.894}, right{5.99242, 6.19633, 46.0950};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({5.99924, 19.5975, 460.894}));
}
TEST_F(TestExact, TestBlastFromLeft) {
  State left{1.0, 0.0, 1000}, right{1.0, 0.0, 0.01};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.575062, 19.59745, 460.8938}));
}
TEST_F(TestExact, TestBlastFromRight) {
  State left{1.0, 0.0, 0.01}, right{1.0, 0.0, 100};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.575113, -6.196328, 46.09504}));
}
TEST_F(TestExact, TestAlmostVacuumed) {
  State left{1.0, -2.0, 0.4}, right{1.0, +2.0, 0.4};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.21852, 0.0, 0.001894}));
}
TEST_F(TestExact, TestVacuumed) {
  State left{1.0, -4.0, 0.4}, right{1.0, +4.0, 0.4};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.0, 0.0, 0.0}));
}

class TestExact2d : public ::testing::Test {
 protected:
  using Solver = Exact<IdealGas<1, 4>, 2>;
  using State = Solver::State;
  using Speed = State::Speed;
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
  State  left{1.0, 0.0, v__left, 1.0};
  State right{1.0, 0.0, v_right, 1.0};
  CompareFlux(solver.GetFlux(left), solver.GetFlux(right));
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux(left));
  left.u() = +1.0;
  right.u() = left.u();
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux(left));
  left.u() = -1.0;
  right.u() = left.u();
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux(right));
}
TEST_F(TestExact2d, TestSod) {
  State  left{1.000, 0.0, v__left, 1.0};
  State right{0.125, 0.0, v_right, 0.1};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.426319, +0.927453, v__left, 0.303130}));
  CompareFlux(solver.GetFluxOnTimeAxis(right, left),
              solver.GetFlux({0.426319, -0.927453, v__left, 0.303130}));
}
TEST_F(TestExact2d, TestShockCollision) {
  State  left{5.99924, 19.59750, v__left, 460.894};
  State right{5.99242, -6.19633, v_right, 46.0950};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({5.99924, 19.5975, v__left, 460.894}));
}
TEST_F(TestExact2d, TestBlastFromLeft) {
  State  left{1.0, 0.0, v__left, 1e+3};
  State right{1.0, 0.0, v_right, 1e-2};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.575062, 19.59745, v__left, 460.8938}));
}
TEST_F(TestExact2d, TestBlastFromRight) {
  State  left{1.0, 0.0, v__left, 1e-2};
  State right{1.0, 0.0, v_right, 1e+2};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.575113, -6.196328, v_right, 46.09504}));
}
TEST_F(TestExact2d, TestAlmostVacuumed) {
  State  left{1.0, -2.0, v__left, 0.4};
  State right{1.0, +2.0, v_right, 0.4};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.21852, 0.0, v__left, 0.001894}));
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.21852, 0.0, v_right, 0.001894}));
}
TEST_F(TestExact2d, TestVacuumed) {
  State  left{1.0, -4.0, v__left, 0.4};
  State right{1.0, +4.0, v_right, 0.4};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.0, 0.0, v__left, 0.0}));
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.0, 0.0, v_right, 0.0}));
}

class TestExact3d : public ::testing::Test {
 protected:
  using Solver = Exact<IdealGas<1, 4>, 3>;
  using State = Solver::State;
  using Speed = State::Speed;
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
  State  left{1.0, 0.0, v__left, w__left, 1.0};
  State right{1.0, 0.0, v_right, w_right, 1.0};
  CompareFlux(solver.GetFlux(left), solver.GetFlux(right));
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux(left));
  left.u() = +1.0;
  right.u() = left.u();
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux(left));
  left.u() = -1.0;
  right.u() = left.u();
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux(right));
}
TEST_F(TestExact3d, TestSod) {
  State  left{1.000, 0.0, v__left, w__left, 1.0};
  State right{0.125, 0.0, v_right, w_right, 0.1};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
      solver.GetFlux({0.426319, +0.927453, v__left, w__left, 0.303130}));
  CompareFlux(solver.GetFluxOnTimeAxis(right, left),
      solver.GetFlux({0.426319, -0.927453, v__left, w__left, 0.303130}));
}
TEST_F(TestExact3d, TestShockCollision) {
  State  left{5.99924, 19.59750, v__left, w__left, 460.894};
  State right{5.99242, -6.19633, v_right, w_right, 46.0950};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({5.99924, 19.5975, v__left, w__left, 460.894}));
}
TEST_F(TestExact3d, TestBlastFromLeft) {
  State  left{1.0, 0.0, v__left, w__left, 1e+3};
  State right{1.0, 0.0, v_right, w_right, 1e-2};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.575062, 19.59745, v__left, w__left, 460.8938}));
}
TEST_F(TestExact3d, TestBlastFromRight) {
  State  left{1.0, 0.0, v__left, w__left, 1e-2};
  State right{1.0, 0.0, v_right, w_right, 1e+2};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
      solver.GetFlux({0.575113, -6.196328, v_right, w_right, 46.09504}));
}
TEST_F(TestExact3d, TestAlmostVacuumed) {
  State  left{1.0, -2.0, v__left, w__left, 0.4};
  State right{1.0, +2.0, v_right, w_right, 0.4};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.21852, 0.0, v__left, w__left, 0.001894}));
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.21852, 0.0, v_right, w_right, 0.001894}));
}
TEST_F(TestExact3d, TestVacuumed) {
  State  left{1.0, -4.0, v__left, w__left, 0.4};
  State right{1.0, +4.0, v_right, w_right, 0.4};
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.0, 0.0, v__left, w__left, 0.0}));
  CompareFlux(solver.GetFluxOnTimeAxis(left, right),
              solver.GetFlux({0.0, 0.0, v_right, w_right, 0.0}));
}

}  // namespace euler
}  // namespace riemann
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
