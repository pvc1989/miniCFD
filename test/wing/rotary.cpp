// Copyright 2022 PEI Weicheng
#include "mini/wing/rotary.hpp"
#include "mini/algebra/eigen.hpp"
#include "mini/geometry/frame.hpp"

#include "gtest/gtest.h"

class TestRotaryWing : public ::testing::Test {
 protected:
  using Scalar = double;
  using Point = mini::algebra::Matrix<Scalar, 3, 1>;
};
TEST_F(TestRotaryWing, Constructors) {
  auto rotor = mini::wing::Rotor<Scalar>();
  rotor.SetOmega(10.0/* rps */);
  rotor.SetOrigin(0.1, 0.2, 0.3);
  auto frame = mini::geometry::Frame<Scalar>();
  frame.RotateY(-0/* deg */);
  rotor.SetFrame(frame);
  // build a blade
  auto airfoil = mini::wing::Airfoil<Scalar>();
  auto blade = mini::wing::Blade<Scalar>();
  Scalar position{0.0}, chord{0.1}, twist{0.0/* deg */};
  blade.InstallSection(position, chord, twist, airfoil);
  position = 2.0, twist = -5.0/* deg */;
  blade.InstallSection(position, chord, twist, airfoil);
  EXPECT_DOUBLE_EQ(blade.GetSpan(), 2.0);
  // install two blades
  Scalar root{0.1};
  auto tip = position + root;
  EXPECT_EQ(rotor.CountBlades(), 0);
  rotor.InstallBlade(root, blade);
  EXPECT_EQ(rotor.CountBlades(), 1);
  rotor.InstallBlade(root, blade);
  EXPECT_EQ(rotor.CountBlades(), 2);
  // test azimuth query
  Scalar deg = 90.0;
  rotor.SetAzimuth(deg);
  EXPECT_DOUBLE_EQ(rotor.GetAzimuth(), deg);
  // test position query
  auto& blade_1 = rotor.GetBlade(1);
  EXPECT_DOUBLE_EQ(blade_1.GetAzimuth(), deg + 180);
  auto rotor_x = rotor.GetFrame().X();
  auto rotor_o = rotor.GetOrigin();
  Point point = rotor_o + rotor_x * root;
  EXPECT_NEAR((blade_1.GetPoint(0.0) - point).norm(), 0, 1e-16);
  point = rotor_o + rotor_x * tip;
  EXPECT_NEAR((blade_1.GetPoint(1.0) - point).norm(), 0, 1e-15);
  // test section query
  auto section = blade_1.GetSection(0.5);
  EXPECT_EQ(section.GetOrigin(), blade_1.GetPoint(0.5));
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
