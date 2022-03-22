// Copyright 2022 PEI Weicheng
#include "mini/wing/rotary.hpp"
#include "mini/geometry/frame.hpp"

#include "gtest/gtest.h"

class TestRotaryWing : public ::testing::Test {
 protected:
  using Scalar = double;
};
TEST_F(TestRotaryWing, Constructors) {
  auto rotor = mini::wing::Rotor<Scalar, 2>();
  rotor.SetOmega(10.0/* rps */);
  rotor.SetOrigin(0, 0, 0);
  auto frame = mini::geometry::Frame<Scalar>();
  frame.RotateY(-5/* deg */);
  rotor.SetOrientation(frame);
  // build a blade
  auto airfoil = mini::wing::Airfoil<Scalar>();
  auto blade = mini::wing::Blade<Scalar>();
  Scalar position{0.0}, chord{0.1}, twist{0.0/* deg */};
  blade.InstallSection(position, chord, twist, airfoil);
  position = 2.0, twist = -5.0/* deg */;
  blade.InstallSection(position, chord, twist, airfoil);
  // install two blades
  Scalar root{0.1};
  rotor.InstallBlade(0, root, blade);
  rotor.InstallBlade(1, root, blade);
  // test azimuth query
  Scalar deg = 90.0;
  Scalar sec = deg / rotor.GetOmega();
  EXPECT_DOUBLE_EQ(rotor.GetAzimuth(sec), deg);
  EXPECT_DOUBLE_EQ(rotor.GetAzimuth(sec, 0), deg);
  EXPECT_DOUBLE_EQ(rotor.GetAzimuth(sec, 1), deg + 180);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
