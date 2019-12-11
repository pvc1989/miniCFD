// Copyright 2019 Weicheng Pei and Minghao Yang
#include "mini/geometry/rectangle.hpp"

#include <vector>

#include "gtest/gtest.h"

namespace mini {
namespace geometry {

class Rectangle2Test : public ::testing::Test {
 protected:
  using Real = double;
  using Rectangle = Rectangle<Real, 2>;
  using Point = Rectangle::Point;
  Point a{0.0, 0.0}, b{1.0, 0.0}, c{1.0, 1.0}, d{0.0, 1.0};
};
TEST_F(Rectangle2Test, Constructor) {
  // Test Rectangle(Point*, Point*, Point*):
  auto rectangle = Rectangle(&a, &b, &c, &d);
  EXPECT_EQ(rectangle.CountVertices(), 4);
}
TEST_F(Rectangle2Test, geometry) {
  auto rectangle = Rectangle(&a, &b, &c, &d);
  EXPECT_EQ(rectangle.Measure(), 1.0);
  auto center = rectangle.Center();
  EXPECT_EQ(center.X() * 4, a.X() + b.X() + c.X() + d.X());
  EXPECT_EQ(center.Y() * 4, a.Y() + b.Y() + c.Y() + d.Y());
}
class Rectangle3Test : public ::testing::Test {
 protected:
  using Real = double;
  using Rectangle = Rectangle<Real, 3>;
  using Point = Rectangle::Point;
  Point a{0.0, 0.0, 0.0}, b{1.0, 0.0, 0.0}, c{1.0, 1.0, 0.0}, d{0.0, 1.0, 0.0};
};
TEST_F(Rectangle3Test, Constructor) {
  // Test Rectangle(Point*, Point*, Point*):
  auto rectangle = Rectangle(&a, &b, &c, &d);
  EXPECT_EQ(rectangle.CountVertices(), 4);
}
TEST_F(Rectangle3Test, geometry) {
  auto rectangle = Rectangle(&a, &b, &c, &d);
  EXPECT_EQ(rectangle.Measure(), 1.0);
  auto center = rectangle.Center();
  EXPECT_EQ(center.X() * 4, a.X() + b.X() + c.X() + d.X());
  EXPECT_EQ(center.Y() * 4, a.Y() + b.Y() + c.Y() + d.Y());
}

}  // namespace geometry
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
