// Copyright 2019 Weicheng Pei and Minghao Yang
#include "mini/element/triangle.hpp"

#include <vector>

#include "gtest/gtest.h"

namespace mini {
namespace element {

class TriangleTest : public ::testing::Test {
 protected:
  using Real = double;
  using Surface = Triangle<Real, 3>;
  using Point = Surface::Point;
  const int i{8};
  Point a{0.0, 0.0, 0.0}, b{1.0, 0.0, 0.0}, c{0.0, 1.0, 0.0};
};
TEST_F(TriangleTest, ConstructorWithId) {
    // Test Triangle(Id, Point*, Point*, Point*):
    auto face = Surface(i, a, b, c);
    EXPECT_EQ(face.CountVertices(), 3);
    EXPECT_EQ(face.I(), i);
}
TEST_F(TriangleTest, ConstructorWithoutId) {
  // Test Triangle(Point*, Point*, Point*):
  auto face = Surface(a, b, c);
  EXPECT_EQ(face.CountVertices(), 3);
  EXPECT_EQ(face.I(), Surface::DefaultId());
}
TEST_F(TriangleTest, MeshMethods) {
  auto face = Surface(a, b, c);
  EXPECT_EQ(face.Measure(), 0.5);
  auto center = face.Center();
  EXPECT_EQ(center.X() * 3, a.X() + b.X() + c.X());
  EXPECT_EQ(center.Y() * 3, a.Y() + b.Y() + c.Y());
  EXPECT_EQ(center.Z() * 3, a.Z() + b.Z() + c.Z());
  auto integrand = [](const auto& point) { return 3.14; };
  EXPECT_EQ(face.Integrate(integrand), face.Measure() * 3.14);
}

}  // namespace element
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
