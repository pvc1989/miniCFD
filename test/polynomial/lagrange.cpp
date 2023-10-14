//  Copyright 2023 PEI Weicheng

#include <cstdlib>

#include "mini/polynomial/lagrange.hpp"

#include "gtest/gtest.h"

double rand_f() {
  return -1 + 2 * std::rand() / (1.0 + RAND_MAX);
}

class TestPolynomialLagrangeLine : public ::testing::Test {
 protected:
  using Scalar = double;
  using Lagrange = mini::polynomial::lagrange::Line<Scalar, 4>;
  static_assert(Lagrange::P == 4);
  static_assert(Lagrange::N == 5);

  static Lagrange GetRandomLagrange() {
    std::srand(31415926);
    return Lagrange{ rand_f(), rand_f(), rand_f(), rand_f(), rand_f() };
  }
  static Lagrange GetUniformLagrange() {
    return Lagrange{ -1, -0.5, 0, 0.5 , 1 };
  }
};
TEST_F(TestPolynomialLagrangeLine, KroneckerDeltaProperty) {
  auto lagrange = GetRandomLagrange();
  for (int i = 0; i < Lagrange::N; ++i) {
    auto x_i = lagrange.GetNode(i);
    auto values = lagrange.GetValues(x_i);
    for (int j = 0; j < Lagrange::N; ++j) {
      EXPECT_EQ(values[j], i == j);
    }
  }
}
TEST_F(TestPolynomialLagrangeLine, PartitionOfUnityProperty) {
  auto lagrange = GetRandomLagrange();
  for (int i = 1<<10; i >= 0; --i) {
    auto x = rand_f();
    auto values = lagrange.GetValues(x);
    EXPECT_NEAR(values.sum(), 1.0, 1e-14);
  }
}
TEST_F(TestPolynomialLagrangeLine, GetDerivatives) {
  // auto lagrange = GetUniformLagrange();
  // std::srand(31415926);
  auto lagrange = GetRandomLagrange();
  auto delta = 1e-5;
  auto delta2 = delta * delta;
  for (int i = 1<<10; i >= 0; --i) {
    auto x = rand_f();
    auto values_center = lagrange.GetValues(x);
    auto derivatives = lagrange.GetDerivatives(x, 0);
    derivatives -= values_center;
    EXPECT_NEAR(derivatives.norm(), 0.0, 1e-11);
    auto values_x_minus = lagrange.GetValues(x - delta);
    auto values_x_plus = lagrange.GetValues(x + delta);
    derivatives = lagrange.GetDerivatives(x, 1);
    derivatives -= (values_x_plus - values_x_minus) / (2 * delta);
    EXPECT_NEAR(derivatives.norm(), 0.0, 1e-7);
    derivatives = lagrange.GetDerivatives(x, 2);
    derivatives -= ((values_x_minus + values_x_plus) - 2 * values_center) / delta2;
    EXPECT_NEAR(derivatives.norm(), 0.0, 1e-4);
  }
}

class TestPolynomialLagrangeHexahedron : public ::testing::Test {
 protected:
  using Scalar = double;
  using Lagrange = mini::polynomial::lagrange::Hexahedron<Scalar, 2, 3, 4>;
  static_assert(Lagrange::N == 3 * 4 * 5);

  static Lagrange GetRandomLagrange() {
    std::srand(31415926);
    return Lagrange(
        Lagrange::LineX{ rand_f(), rand_f(), rand_f() },
        Lagrange::LineY{ rand_f(), rand_f(), rand_f(), rand_f() },
        Lagrange::LineZ{ rand_f(), rand_f(), rand_f(), rand_f(), rand_f() }
    );
  }
};
TEST_F(TestPolynomialLagrangeHexahedron, KroneckerDeltaProperty) {
  auto lagrange = GetRandomLagrange();
  for (int i = 0; i < Lagrange::N; ++i) {
    auto values = lagrange.GetValues(lagrange.GetNode(i));
    for (int j = 0; j < Lagrange::N; ++j) {
      EXPECT_EQ(values[j], i == j);
    }
  }
}
TEST_F(TestPolynomialLagrangeHexahedron, PartitionOfUnityProperty) {
  auto lagrange = GetRandomLagrange();
  for (int i = 1<<10; i >= 0; --i) {
    auto x = rand_f(), y = rand_f(), z = rand_f();
    auto values = lagrange.GetValues(x, y, z);
    EXPECT_NEAR(values.sum(), 1.0, 1e-12);
  }
}
TEST_F(TestPolynomialLagrangeHexahedron, GetDerivatives) {
  // auto lagrange = GetUniformLagrange();
  // std::srand(31415926);
  auto lagrange = GetRandomLagrange();
  auto delta = 1e-5;
  auto delta2 = delta * delta;
  for (int i = 1<<10; i >= 0; --i) {
    auto x = rand_f(), y = rand_f(), z = rand_f();
    auto values_center = lagrange.GetValues(x, y, z);
    auto derivatives = lagrange.GetDerivatives(x, y, z, 0, 0, 0);
    derivatives -= values_center;
    EXPECT_NEAR(derivatives.norm(), 0.0, 1e-10);
    auto values_x_minus = lagrange.GetValues(x - delta, y, z);
    auto values_x_plus = lagrange.GetValues(x + delta, y, z);
    derivatives = lagrange.GetDerivatives(x, y, z, 1, 0, 0);
    derivatives -= (values_x_plus - values_x_minus) / (2 * delta);
    EXPECT_NEAR(derivatives.norm(), 0.0, 1e-7);
    auto values_y_minus = lagrange.GetValues(x, y - delta, z);
    auto values_y_plus = lagrange.GetValues(x, y + delta, z);
    derivatives = lagrange.GetDerivatives(x, y, z, 0, 1, 0);
    derivatives -= (values_y_plus - values_y_minus) / (2 * delta);
    EXPECT_NEAR(derivatives.norm(), 0.0, 1e-6);
    auto values_z_minus = lagrange.GetValues(x, y, z - delta);
    auto values_z_plus = lagrange.GetValues(x, y, z + delta);
    derivatives = lagrange.GetDerivatives(x, y, z, 0, 0, 1);
    derivatives -= (values_z_plus - values_z_minus) / (2 * delta);
    EXPECT_NEAR(derivatives.norm(), 0.0, 1e-5);
    derivatives = lagrange.GetDerivatives(x, y, z, 2, 0, 0);
    derivatives -= ((values_x_minus + values_x_plus) - 2 * values_center) / delta2;
    EXPECT_NEAR(derivatives.norm(), 0.0, 1e-2);
    derivatives = lagrange.GetDerivatives(x, y, z, 0, 2, 0);
    derivatives -= ((values_y_plus + values_y_minus) - 2 * values_center) / delta2;
    EXPECT_NEAR(derivatives.norm(), 0.0, 1e-2);
    derivatives = lagrange.GetDerivatives(x, y, z, 0, 0, 2);
    derivatives -= ((values_z_plus + values_z_minus) - 2 * values_center) / delta2;
    EXPECT_NEAR(derivatives.norm(), 0.0, 1e-2);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
