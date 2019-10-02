// Copyright 2019 Weicheng Pei and Minghao Yang

#include <vector>

#include "gtest/gtest.h"

#include "mini/riemann/linear/double.hpp"

namespace mini {
namespace riemann {
namespace linear {

class DoubleWaveTest : public ::testing::Test {
 protected:
  using Solver = Double;
  using State = Solver::State;
  using Flux = Solver::Flux;
  using Column = Solver::Column;
  using Matrix = Solver::Matrix;
  State left{1.0, 11.0}, right{2.0, 22.0};
};
TEST_F(DoubleWaveTest, TestTwoLeftRunningWaves) {
  // eigen_values = {-2, -1}
  auto solver = Solver(Matrix{{-2.0, 0.0}, {0.0, -1.0}});
  auto f_on_t_axia = solver.GetFluxOnTimeAxis(left, right);
  EXPECT_DOUBLE_EQ(f_on_t_axia[0], -4.0);
  EXPECT_DOUBLE_EQ(f_on_t_axia[1], -22.0);
}
TEST_F(DoubleWaveTest, TestTwoRightRunningWaves) {
  // eigen_values = {1, 2}
  auto solver = Solver(Matrix{{1.0, 0.0}, {0.0, 2.0}});
  auto f_on_t_axia = solver.GetFluxOnTimeAxis(left, right);
  EXPECT_DOUBLE_EQ(f_on_t_axia[0], 1.0);
  EXPECT_DOUBLE_EQ(f_on_t_axia[1], 22.0);
}
TEST_F(DoubleWaveTest, TestBetweenTwoWaves) {
  // eigen_values = {-1, 1}
  auto solver = Solver(Matrix{{-1.0, 0.0}, {0.0, +1.0}});
  auto f_on_t_axia = solver.GetFluxOnTimeAxis(left, right);
  EXPECT_DOUBLE_EQ(f_on_t_axia[0], -2.0);
  EXPECT_DOUBLE_EQ(f_on_t_axia[1], 11.0);
}
TEST_F(DoubleWaveTest, TestNonTrivialMatrix) {
  // eigen_values = {1, -1}
  auto solver = Solver(Matrix{{-5.0, 4.0}, {-4.0, 5.0}});
  auto f_on_t_axia = solver.GetFluxOnTimeAxis(left, right);
  EXPECT_DOUBLE_EQ(f_on_t_axia[0], 57.0);
  EXPECT_DOUBLE_EQ(f_on_t_axia[1], 60.0);
}

}  // namespace linear
}  // namespace riemann
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
