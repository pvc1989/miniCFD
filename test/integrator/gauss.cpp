// Copyright 2019 Weicheng Pei and Minghao Yang
#include "mini/integrator/gauss.hpp"

#include <cmath>

#include "gtest/gtest.h"


namespace mini {
namespace integrator {

class GaussTest : public ::testing::Test {
 protected:
  double MonomialIntegral(int p) {
    // \int_{-1}^{+1} x^{p} dx
    return p % 2 ? 0.0 : 2.0 / (p + 1);
  }
  double eps = 1e-15;
};
TEST_F(GaussTest, OnePoint) {
  constexpr int kPoints = 1;
  constexpr int kDegree = kPoints * 2 - 1;
  for (int p = 0; p != kDegree; ++p) {
    auto integrand = [&](double x) { return std::pow(x, p); };
    EXPECT_NEAR(Gauss<kPoints>::Integrate(integrand), MonomialIntegral(p), eps);
  }
}
TEST_F(GaussTest, TwoPoint) {
  constexpr int kPoints = 2;
  constexpr int kDegree = kPoints * 2 - 1;
  for (int p = 0; p != kDegree; ++p) {
    auto integrand = [&](double x) { return std::pow(x, p); };
    EXPECT_NEAR(Gauss<kPoints>::Integrate(integrand), MonomialIntegral(p), eps);
  }
}
TEST_F(GaussTest, ThreePoint) {
  constexpr int kPoints = 3;
  constexpr int kDegree = kPoints * 2 - 1;
  for (int p = 0; p != kDegree; ++p) {
    auto integrand = [&](double x) { return std::pow(x, p); };
    EXPECT_NEAR(Gauss<kPoints>::Integrate(integrand), MonomialIntegral(p), eps);
  }
}
TEST_F(GaussTest, FourPoint) {
  constexpr int kPoints = 4;
  constexpr int kDegree = kPoints * 2 - 1;
  for (int p = 0; p != kDegree; ++p) {
    auto integrand = [&](double x) { return std::pow(x, p); };
    EXPECT_NEAR(Gauss<kPoints>::Integrate(integrand), MonomialIntegral(p), eps);
  }
}
TEST_F(GaussTest, FivePoint) {
  constexpr int kPoints = 5;
  constexpr int kDegree = kPoints * 2 - 1;
  for (int p = 0; p != kDegree; ++p) {
    auto integrand = [&](double x) { return std::pow(x, p); };
    EXPECT_NEAR(Gauss<kPoints>::Integrate(integrand), MonomialIntegral(p), eps);
  }
}
TEST_F(GaussTest, TwoPoint) {
  TestScalarFunction<2>();
  TestVectorFunction<2>();
}
TEST_F(GaussTest, ThreePoint) {
  TestScalarFunction<3>();
  TestVectorFunction<3>();
}
TEST_F(GaussTest, FourPoint) {
  TestScalarFunction<4>();
  TestVectorFunction<4>();
}
TEST_F(GaussTest, FivePoint) {
  TestScalarFunction<5>();
  TestVectorFunction<5>();
}

}  // namespace integrator
}  // namespace mini
