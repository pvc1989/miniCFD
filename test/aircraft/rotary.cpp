// Copyright 2022 PEI Weicheng
#include <vector>

#include "mini/aircraft/airfoil.hpp"
#include "mini/aircraft/rotor.hpp"
#include "mini/algebra/eigen.hpp"
#include "mini/geometry/frame.hpp"

#include "gtest/gtest.h"


class TestRotaryWing : public ::testing::Test {
 protected:
  using Scalar = double;
  using Vector = mini::algebra::Matrix<Scalar, 3, 1>;
  using Point = Vector;
};
TEST_F(TestRotaryWing, Constructors) {
  auto rotor = mini::aircraft::Rotor<Scalar>();
  rotor.SetRevolutionsPerSecond(20.0);
  auto omega = mini::geometry::pi() * 40;
  EXPECT_DOUBLE_EQ(rotor.GetOmega(), omega);
  rotor.SetOmega(omega);
  EXPECT_DOUBLE_EQ(rotor.GetOmega(), omega);
  rotor.SetOrigin(0.1, 0.2, 0.3);
  auto frame = mini::geometry::Frame<Scalar>();
  frame.RotateY(-5/* deg */);
  rotor.SetFrame(frame);
  // build a blade
  auto blade = mini::aircraft::Blade<Scalar>();
  auto airfoils = std::vector<mini::aircraft::airfoil::Simple<Scalar>>();
  airfoils.emplace_back(6.0, 0.0);
  airfoils.emplace_back(5.0, 0.2);
  std::vector<Scalar> y_values{0.0, 2.0}, chords{0.3, 0.1}, twists{0.0, -5.0};
  blade.InstallSection(y_values[0], chords[0], twists[0], airfoils[0]);
  blade.InstallSection(y_values[1], chords[1], twists[1], airfoils[1]);
  EXPECT_DOUBLE_EQ(blade.GetSpan(), y_values.back());
  // test section query
  auto section = blade.GetSection(0.5);
  EXPECT_DOUBLE_EQ(section.GetChord(), +0.2);
  EXPECT_DOUBLE_EQ(section.GetTwist(), -2.5);
  EXPECT_DOUBLE_EQ(section.Lift(12.5), 5.5);
  EXPECT_DOUBLE_EQ(section.Drag(12.5), 0.1);
  // install two blades
  Scalar root{0.1};
  EXPECT_EQ(rotor.CountBlades(), 0);
  rotor.InstallBlade(root, blade);
  EXPECT_EQ(rotor.CountBlades(), 1);
  rotor.InstallBlade(root, blade);
  EXPECT_EQ(rotor.CountBlades(), 2);
}
TEST_F(TestRotaryWing, NightyDegree) {
  auto rotor = mini::aircraft::Rotor<Scalar>();
  rotor.SetRevolutionsPerSecond(20.0);
  auto omega = mini::geometry::pi() * 40;
  rotor.SetOrigin(0.1, 0.2, 0.3);
  auto frame = mini::geometry::Frame<Scalar>();
  frame.RotateY(-5/* deg */);
  rotor.SetFrame(frame);
  // build a blade
  auto blade = mini::aircraft::Blade<Scalar>();
  auto airfoils = std::vector<mini::aircraft::airfoil::Simple<Scalar>>();
  airfoils.emplace_back(6.0, 0.0);
  airfoils.emplace_back(5.0, 0.2);
  std::vector<Scalar> y_values{0.0, 2.0}, chords{0.3, 0.1}, twists{0.0, -5.0};
  blade.InstallSection(y_values[0], chords[0], twists[0], airfoils[0]);
  blade.InstallSection(y_values[1], chords[1], twists[1], airfoils[1]);
  EXPECT_DOUBLE_EQ(blade.GetSpan(), y_values.back());
  // install two blades
  Scalar root{0.1};
  rotor.InstallBlade(root, blade);
  rotor.InstallBlade(root, blade);
  // test azimuth query
  Scalar deg = 90.0;
  rotor.SetAzimuth(deg);
  EXPECT_DOUBLE_EQ(rotor.GetAzimuth(), deg);
  // test y_value query
  auto& blade_1 = rotor.GetBlade(1);  // blade_y == rotor_x
  EXPECT_DOUBLE_EQ(blade_1.GetAzimuth(), deg + 180);
  auto rotor_x = rotor.GetFrame().X();
  auto rotor_o = rotor.GetOrigin();
  Point point = rotor_o + rotor_x * root;
  EXPECT_NEAR((blade_1.GetPoint(0.0) - point).norm(), 0, 1e-16);
  auto tip = root + blade_1.GetSpan();
  point = rotor_o + rotor_x * tip;
  EXPECT_NEAR((blade_1.GetPoint(1.0) - point).norm(), 0, 1e-15);
  // test section query
  auto section = blade_1.GetSection(0.5);
  EXPECT_DOUBLE_EQ(section.GetChord(), +0.2);
  EXPECT_DOUBLE_EQ(section.GetTwist(), -2.5);
  EXPECT_DOUBLE_EQ(section.Lift(12.5), 5.5);
  EXPECT_DOUBLE_EQ(section.Drag(12.5), 0.1);
  EXPECT_EQ(section.GetOrigin(), blade_1.GetPoint(0.5));
  auto v_norm = rotor.GetOmega() * (root + tip) / 2;
  auto veclocity = -v_norm * blade_1.GetFrame().X();
  EXPECT_NEAR(veclocity.dot(rotor.GetFrame().X()), 0.0, 1e-13);
  EXPECT_NEAR((section.GetVelocity() - veclocity).norm(), 0, 1e-13);
}
TEST_F(TestRotaryWing, HalfCycle) {
  auto rotor = mini::aircraft::Rotor<Scalar>();
  rotor.SetRevolutionsPerSecond(20.0);
  auto omega = mini::geometry::pi() * 40;
  rotor.SetOrigin(0.1, 0.2, 0.3);
  auto frame = mini::geometry::Frame<Scalar>();
  frame.RotateY(-5/* deg */);
  rotor.SetFrame(frame);
  // build a blade
  auto blade = mini::aircraft::Blade<Scalar>();
  auto airfoils = std::vector<mini::aircraft::airfoil::Simple<Scalar>>();
  airfoils.emplace_back(6.0, 0.0);
  airfoils.emplace_back(5.0, 0.2);
  std::vector<Scalar> y_values{0.0, 2.0}, chords{0.3, 0.1}, twists{0.0, -5.0};
  blade.InstallSection(y_values[0], chords[0], twists[0], airfoils[0]);
  blade.InstallSection(y_values[1], chords[1], twists[1], airfoils[1]);
  EXPECT_DOUBLE_EQ(blade.GetSpan(), y_values.back());
  // install two blades
  Scalar root{0.1};
  rotor.InstallBlade(root, blade);
  rotor.InstallBlade(root, blade);
  // test azimuth query
  Scalar deg = 180.0;
  rotor.SetAzimuth(deg);
  EXPECT_DOUBLE_EQ(rotor.GetAzimuth(), deg);
  // test y_value query
  auto& blade_0 = rotor.GetBlade(0);  // blade_y == -rotor_y
  EXPECT_DOUBLE_EQ(blade_0.GetAzimuth(), deg);
  auto rotor_y = rotor.GetFrame().Y();
  auto rotor_o = rotor.GetOrigin();
  Point point = rotor_o - rotor_y * root;
  EXPECT_NEAR((blade_0.GetPoint(0.0) - point).norm(), 0, 1e-16);
  auto tip = root + blade_0.GetSpan();
  point = rotor_o - rotor_y * tip;
  EXPECT_NEAR((blade_0.GetPoint(1.0) - point).norm(), 0, 1e-15);
  // test section query
  auto section = blade_0.GetSection(0.5);
  EXPECT_DOUBLE_EQ(section.GetChord(), +0.2);
  EXPECT_DOUBLE_EQ(section.GetTwist(), -2.5);
  EXPECT_DOUBLE_EQ(section.Lift(12.5), 5.5);
  EXPECT_DOUBLE_EQ(section.Drag(12.5), 0.1);
  EXPECT_EQ(section.GetOrigin(), blade_0.GetPoint(0.5));
  auto v_norm = rotor.GetOmega() * (root + tip) / 2;
  auto veclocity = -v_norm * blade_0.GetFrame().X();
  EXPECT_NEAR(veclocity.dot(rotor.GetFrame().Y()), 0.0, 1e-13);
  EXPECT_NEAR((section.GetVelocity() - veclocity).norm(), 0, 1e-13);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
