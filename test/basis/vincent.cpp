//  Copyright 2023 PEI Weicheng

#include <cmath>
#include <cstdlib>

#include "mini/basis/vincent.hpp"
#include "mini/gauss/line.hpp"
#include "mini/gauss/function.hpp"

#include "gtest/gtest.h"

class TestBasisVincent : public ::testing::Test {
 protected:
  using Scalar = double;
  using Vincent = mini::basis::Vincent<Scalar>;
};
TEST_F(TestBasisVincent, DiscontinuousGalerkin) {
  auto line_gauss = mini::gauss::Line<Scalar, 1, 6>(-1, 1);
  for (int degree = 1; degree < 6; ++degree) {
    auto vincent = Vincent(degree, Vincent::DiscontinuousGalerkin(degree));
    // check values at ends
    EXPECT_EQ(vincent.LocalToRightValue(+1.0), 1.0);
    EXPECT_EQ(vincent.LocalToRightValue(-1.0), 0.0);
    EXPECT_EQ(vincent.LocalToLeftValue(+1.0), 0.0);
    EXPECT_EQ(vincent.LocalToLeftValue(-1.0), 1.0);
    // check orthogonality with P_{N - 2 == degree - 1}
    for (int l = 0; l < degree - 0; ++l) {
      auto ip = mini::gauss::Innerprod(
          [&vincent](Scalar x){ return vincent.LocalToRightValue(x); },
          [l](Scalar x){ return std::legendre(l, x); },
          line_gauss);
      EXPECT_NEAR(ip, 0, 1e-15);
    }
  }
}
TEST_F(TestBasisVincent, HuynhLumpingLobatto) {
  auto line_gauss = mini::gauss::Line<Scalar, 1, 5>(-1, 1);
  for (int degree = 1; degree < 6; ++degree) {
    auto vincent = Vincent(degree, Vincent::HuynhLumpingLobatto(degree));
    // check values at ends
    EXPECT_EQ(vincent.LocalToRightValue(+1.0), 1.0);
    EXPECT_NEAR(vincent.LocalToRightValue(-1.0), 0.0, 1e-16);
    EXPECT_NEAR(vincent.LocalToLeftValue(+1.0), 0.0, 1e-16);
    EXPECT_EQ(vincent.LocalToLeftValue(-1.0), 1.0);
    // check orthogonality with P_{N - 3 == degree - 2}
    for (int l = 0; l < degree - 1; ++l) {
      auto ip = mini::gauss::Innerprod(
          [&vincent](Scalar x){ return vincent.LocalToRightValue(x); },
          [l](Scalar x){ return std::legendre(l, x); },
          line_gauss);
      EXPECT_NEAR(ip, 0, 1e-15);
    }
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
