// Copyright 2019 Weicheng Pei and Minghao Yang
#include "mini/integrator/gauss.hpp"
#include "mini/algebra/column.hpp"

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
  template <int kPoints>
  void TestVectorFunction() {
    auto integrand = [&](double x) {
      auto result = algebra::Column<double, 2 * kPoints>();
      for (int p = 0; p != 2 * kPoints; ++p) {
        result[p] = std::pow(x, p);
      }
      return result;
    };
    auto result = Gauss<kPoints>::Integrate(integrand);
    for (int p = 0; p != 2 * kPoints; ++p) {
      EXPECT_DOUBLE_EQ(result[p], MonomialIntegral(p));
    }
  }
};
TEST_F(GaussTest, OnePoint) {
  constexpr int kDegree = 1;
  for (int p = 0; p != kDegree; ++p) {
    auto integrand = [&](double x) { return std::pow(x, p); };
    EXPECT_DOUBLE_EQ(Gauss<kDegree>::Integrate(integrand), MonomialIntegral(p));
  }
  TestVectorFunction<1>();
}

}  // namespace integrator
}  // namespace mini
