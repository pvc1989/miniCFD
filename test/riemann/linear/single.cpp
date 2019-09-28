// Copyright 2019 Weicheng Pei and Minghao Yang

#include <vector>

#include "gtest/gtest.h"

#include "mini/riemann/linear/single.hpp"

namespace mini {
namespace riemann {
namespace linear {

class SingleWaveTest : public ::testing::Test {
 protected:
  using Solver = SingleWave;
  using State = Solver::State;
  using Flux = Solver::Flux;
  using Speed = Solver::Speed;
};
TEST_F(SingleWaveTest, TestFlux) {
  State u_l{2.0}, u_r{1.0};
  // right running wave
  auto solver = Solver(/* speed = */1.0);
  EXPECT_EQ(solver.GetFluxOnTimeAxis(u_l, u_r), solver.GetFlux(u_l));
  // left running wave
  solver = Solver(-1.0);
  EXPECT_EQ(solver.GetFluxOnTimeAxis(u_l, u_r), solver.GetFlux(u_r));
  // standing wave
  solver = Solver(0.0);
  EXPECT_EQ(solver.GetFluxOnTimeAxis(u_l, u_r), solver.GetFlux(u_l));
}

}  // namespace linear
}  // namespace riemann
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
