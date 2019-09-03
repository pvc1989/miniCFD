// Copyright 2019 Weicheng Pei and Minghao Yang

#include <vector>

#include "gtest/gtest.h"

#include "mini/riemann/linear.hpp"

namespace mini {
namespace riemann {

class SingleWaveTest : public :: testing::Test {
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

class MultiWaveTest : public :: testing::Test {
 protected:
  using Solver = MultiWave<2>;
  using State = Solver::State;
  using Flux = Solver::Flux;
  using Column = Solver::Column;
  using Matrix = Solver::Matrix;
};
TEST_F(MultiWaveTest, TestFlux) {
  State u_l = {1.0, 11.0};
  State u_r = {2.0, 22.0};
  // eigen_values = {1, -1}
  auto solver = Solver(Matrix{-5.0, 4.0, -4.0, 5.0});
  auto f_on_t_axia = solver.GetFluxOnTimeAxis(u_l, u_r);
  EXPECT_DOUBLE_EQ(f_on_t_axia[0], 57.0);
  EXPECT_DOUBLE_EQ(f_on_t_axia[1], 60.0);
}

}  // namespace riemann
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
