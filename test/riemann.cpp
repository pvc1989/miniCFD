// Copyright 2019 Weicheng Pei and Minghao Yang

#include <vector>

#include "gtest/gtest.h"

#include "mini/riemann/linear.hpp"

namespace mini {
namespace riemann {

class SingleWaveTest : public :: testing::Test {
 protected:
  using State = SingleWave::State;
  using Flux = SingleWave::Flux;
  using Speed = SingleWave::Speed;
};
TEST_F(SingleWaveTest, TestFlux) {
  State u_l{2.0}, u_r{1.0};
  // right running wave
  auto solver = SingleWave(/* speed = */1.0);
  EXPECT_EQ(solver.GetFluxOnTimeAxis(u_l, u_r), solver.GetFlux(u_l));
  // left running wave
  solver = SingleWave(-1.0);
  EXPECT_EQ(solver.GetFluxOnTimeAxis(u_l, u_r), solver.GetFlux(u_r));
  // standing wave
  solver = SingleWave(0.0);
  EXPECT_EQ(solver.GetFluxOnTimeAxis(u_l, u_r), solver.GetFlux(u_l));
}

}  // namespace riemann
}  // namespace mini
