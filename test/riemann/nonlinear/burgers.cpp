// Copyright 2019 PEI Weicheng and YANG Minghao

#include <vector>

#include "gtest/gtest.h"

#include "mini/riemann/nonlinear/burgers.hpp"

namespace mini {
namespace riemann {
namespace nonlinear {

class TestBurgers : public ::testing::Test {
 protected:
  using Solver = Burgers<double, 2>;
  using Conservative = Solver::Conservative;
};
TEST_F(TestBurgers, TestNonZeroK) {
  for (auto k : {+2.0, -2.0}) {
    auto solver = Solver(k);
    // smooth
    Conservative u_l{1.54 / k};
    Conservative u_r{u_l};
    EXPECT_EQ(solver.GetFluxOnTimeAxis(u_l, u_r), solver.GetFlux(u_l));
    EXPECT_EQ(solver.GetFluxOnTimeAxis(u_l, u_r), solver.GetFlux(u_r));
    u_l = -1.54 / k;
    u_r = -1.54 / k;
    EXPECT_EQ(solver.GetFluxOnTimeAxis(u_l, u_r), solver.GetFlux(u_l));
    EXPECT_EQ(solver.GetFluxOnTimeAxis(u_l, u_r), solver.GetFlux(u_r));
    // right running shock
    u_l = 2.0 / k;
    u_r = 1.0 / k;
    EXPECT_EQ(solver.GetFluxOnTimeAxis(u_l, u_r), solver.GetFlux(u_l));
    // left running shock
    u_l = -1.0 / k;
    u_r = -2.0 / k;
    EXPECT_EQ(solver.GetFluxOnTimeAxis(u_l, u_r), solver.GetFlux(u_r));
    // standing shock
    u_l = +1.0 / k;
    u_r = -1.0 / k;
    EXPECT_EQ(solver.GetFluxOnTimeAxis(u_l, u_r), solver.GetFlux(u_l));
    EXPECT_EQ(solver.GetFluxOnTimeAxis(u_l, u_r), solver.GetFlux(u_r));
    // right running expansion
    u_l = +1.0 / k;
    u_r = +2.0 / k;
    EXPECT_EQ(solver.GetFluxOnTimeAxis(u_l, u_r), solver.GetFlux(u_l));
    // left running expansion
    u_l = -2.0 / k;
    u_r = -1.0 / k;
    EXPECT_EQ(solver.GetFluxOnTimeAxis(u_l, u_r), solver.GetFlux(u_r));
    // inside expansion
    u_l = -1.0 / k;
    u_r = +2.0 / k;
    EXPECT_EQ(solver.GetFluxOnTimeAxis(u_l, u_r), solver.GetFlux(0.0));
    u_l = -2.0 / k;
    u_r = +1.0 / k;
    EXPECT_EQ(solver.GetFluxOnTimeAxis(u_l, u_r), solver.GetFlux(0.0));
  }
}

}  // namespace nonlinear
}  // namespace riemann
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
