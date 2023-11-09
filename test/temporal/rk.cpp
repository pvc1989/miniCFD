//  Copyright 2023 PEI Weicheng

#include <cstdlib>

#include "mini/temporal/ode.hpp"
#include "mini/temporal/rk.hpp"

#include "gtest/gtest.h"

class TestTemporalConstant : public ::testing::Test {
 protected:
  using Scalar = float;
  using System = mini::temporal::Constant<Scalar>;
  using Matrix = typename System::Matrix;
  using Column = typename System::Column;

  static double rand_f() {
    return std::rand() / (1.0 + RAND_MAX);
  }
};
TEST_F(TestTemporalConstant, OneStepRungeKutta) {
  using Solver = mini::temporal::RungeKutta<1, Scalar>;
  int n = 10;
  std::srand(31415926);
  Matrix a = Matrix::Random(n, n);
  auto system = System(a);
  Column u_old = Column::Random(n);
  system.SetSolutionColumn(u_old);
  auto solver = Solver();
  auto t_curr = rand_f();
  auto dt = rand_f();
  solver.Update(&system, t_curr, dt);
  auto u_new = system.GetSolutionColumn();
  EXPECT_EQ(u_new, u_old + a * u_old * dt);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}