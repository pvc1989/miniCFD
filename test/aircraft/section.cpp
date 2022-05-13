// Copyright 2022 PEI Weicheng
#include <vector>

#include "mini/aircraft/airfoil.hpp"
#include "mini/aircraft/section.hpp"
#include "mini/aircraft/blade.hpp"
#include "mini/geometry/pi.hpp"

#include "gtest/gtest.h"

class TestBladeSection : public ::testing::Test {
 protected:
  using Scalar = double;
  using Blade = mini::aircraft::Blade<Scalar>;
  using Section = typename Blade::Section;
  using Airfoil = mini::aircraft::airfoil::SC1095<Scalar>;
};
TEST_F(TestBladeSection, Constructors) {
  auto blade = Blade();
  auto airfoil = Airfoil();
  auto y_ratio = 0.2, chord = 1.0, twist = 5.0;
  auto section = Section(blade, y_ratio, chord, twist,
      &airfoil, 0.5, &airfoil, 0.5);
  for (Scalar aoi = -1000; aoi < 1000; aoi += 5) {
    // aoi := angle of inflow
    auto [u, w] = mini::geometry::CosSin(aoi);
    // aoa := angle of attack
    auto aoa_actual = section.GetAngleOfAttack(u, w);
    auto aoa_expect = aoi + twist;
    aoa_actual = mini::geometry::deg2rad(aoa_actual);
    aoa_expect = mini::geometry::deg2rad(aoa_expect);
    EXPECT_NEAR(std::cos(aoa_actual), std::cos(aoa_expect), 1e-9);
    EXPECT_NEAR(std::sin(aoa_actual), std::sin(aoa_expect), 1e-9);
  }
}
