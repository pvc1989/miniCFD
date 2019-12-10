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
};
TEST_F(GaussTest, OnePoint) {
  constexpr int kDegree = 1;
  for (int p = 0; p != kDegree; ++p) {
    auto integrand = [&](double x) { return std::pow(x, p); };
    EXPECT_DOUBLE_EQ(Gauss<kDegree>::Integrate(integrand), MonomialIntegral(p));
  }
}

}  // namespace integrator
}  // namespace mini
