// Copyright 2019 Weicheng Pei and Minghao Yang

#include "mini/geometry/rectangle.hpp"

#include <vector>

#include "gtest/gtest.h"

namespace mini {
namespace geometry {

class TestRectangle2d : public ::testing::Test {
 protected:
  using Real = double;
  using Rectangle = Rectangle<Real, 2>;
  using Point = Rectangle::Point;
  Point a{0.0, 0.0}, b{1.0, 0.0}, c{1.0, 1.0}, d{0.0, 1.0};
};
TEST_F(TestRectangle2d, Constructor) {
  // Test Rectangle(Point const&, Point const&, Point const&):
  auto rectangle = Rectangle(a, b, c, d);
  EXPECT_EQ(rectangle.CountVertices(), 4);
}
TEST_F(TestRectangle2d, GeometricMethods) {
  auto rectangle = Rectangle(a, b, c, d);
  EXPECT_EQ(rectangle.Measure(), 1.0);
  auto center = rectangle.Center();
  EXPECT_EQ(center.X() * 4, a.X() + b.X() + c.X() + d.X());
  EXPECT_EQ(center.Y() * 4, a.Y() + b.Y() + c.Y() + d.Y());
  EXPECT_EQ(center.Z() * 4, a.Z() + b.Z() + c.Z() + d.Z());
}
class TestRectangle3d : public ::testing::Test {
 protected:
  using Real = double;
  using Rectangle = Rectangle<Real, 3>;
  using Point = Rectangle::Point;
  Point a{0.0, 0.0, 0.0}, b{1.0, 0.0, 0.0}, c{1.0, 1.0, 0.0}, d{0.0, 1.0, 0.0};
};
TEST_F(TestRectangle3d, Constructor) {
  // Test Rectangle(Point const&, Point const&, Point const&):
  auto rectangle = Rectangle(a, b, c, d);
  EXPECT_EQ(rectangle.CountVertices(), 4);
}
TEST_F(TestRectangle3d, GeometricMethods) {
  auto rectangle = Rectangle(a, b, c, d);
  EXPECT_EQ(rectangle.Measure(), 1.0);
  auto center = rectangle.Center();
  EXPECT_EQ(center.X() * 4, a.X() + b.X() + c.X() + d.X());
  EXPECT_EQ(center.Y() * 4, a.Y() + b.Y() + c.Y() + d.Y());
  EXPECT_EQ(center.Z() * 4, a.Z() + b.Z() + c.Z() + d.Z());
}

}  // namespace geometry
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
