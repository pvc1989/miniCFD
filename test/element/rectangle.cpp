// Copyright 2019 Weicheng Pei and Minghao Yang
#include "mini/element/rectangle.hpp"

#include <vector>

#include "gtest/gtest.h"

namespace mini {
namespace element {

class RectangleTest : public ::testing::Test {
 protected:
  using Real = double;
  using Surface = Rectangle<Real, 3>;
  using Point = Surface::Point;
  const int i{8};
  Point a{0.0, 0.0, 0.0}, b{1.0, 0.0, 0.0}, c{1.0, 1.0, 0.0}, d{0.0, 1.0, 0.0};
};
TEST_F(RectangleTest, ConstructorWithId) {
    // Test Rectangle(Id, Point*, Point*, Point*, Point*):
    auto face = Surface(i, a, b, c, d);
    EXPECT_EQ(face.CountVertices(), 4);
    EXPECT_EQ(face.I(), i);
}
TEST_F(RectangleTest, ConstructorWithoutId) {
  // Test Rectangle(Point*, Point*, Point*, Point*):
  auto face = Surface(a, b, c, d);
  EXPECT_EQ(face.CountVertices(), 4);
  EXPECT_EQ(face.I(), Surface::DefaultId());
}
TEST_F(RectangleTest, MeshMethods) {
  auto face = Surface(a, b, c, d);
  EXPECT_EQ(face.Measure(), 1.0);
  auto center = face.Center();
  EXPECT_EQ(center.X() * 4, a.X() + b.X() + c.X() + d.X());
  EXPECT_EQ(center.Y() * 4, a.Y() + b.Y() + c.Y() + d.Y());
  EXPECT_EQ(center.Z() * 4, a.Z() + b.Z() + c.Z() + d.Z());
  auto integrand = [](const auto& point) { return 3.14; };
  EXPECT_EQ(face.Integrate(integrand), face.Measure() * 3.14);
}

}  // namespace element
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
