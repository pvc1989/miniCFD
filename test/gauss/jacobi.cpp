// Copyright 2023 PEI Weicheng

#include <cmath>
#include <cstdlib>

#include "mini/gauss/jacobi.hpp"
#include "mini/gauss/function.hpp"

#include "gtest/gtest.h"

class TestJacobi : public ::testing::Test {
 protected:
  static constexpr int kAlpha{2}, kBeta{0};
  static double rand() {
    return std::rand() / (1.0 + RAND_MAX);
  }
  static double f(double *a, int n, double x) {
    double val = 0.0;
    for (int i = 0; i < n; ++i) {
      val += a[i] * std::pow(1 - x, i);
    }
    return val;
  }
  static double exact(double *a, int n) {
    double val = 0.0;
    for (int i = 0; i < n; ++i) {
      val += a[i] * std::pow(2, i + 3) / (i + 3);
    }
    return val;
  }
};
TEST_F(TestJacobi, ThreePoint) {
  using Gauss = mini::gauss::Jacobi<double, 3, kAlpha, kBeta>;
  std::srand(31415926);
  std::vector<double> v = { rand(), rand(), rand(), rand(), rand() };
  auto *a = v.data();
  int n = v.size();
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += f(a, n, Gauss::points[i]) * Gauss::weights[i];
  }
  EXPECT_NEAR(sum, exact(a, n), 1e-13);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
