// Copyright 2019 Weicheng Pei and Minghao Yang
#include "mini/geometry/triangle.hpp"

#include <vector>

#include "gtest/gtest.h"

namespace mini {
namespace geometry {

class Triangle2dTest : public ::testing::Test {
 protected:
  using Real = double;
  using Triangle = Triangle<Real, 2>;
  using Point = Triangle::Point;
  Point a{0.0, 0.0}, b{1.0, 0.0}, c{0.0, 1.0};
};
TEST_F(Triangle2dTest, Constructor) {
  // Test Triangle(Point const&, Point const&, Point const&):
  auto triangle = Triangle(a, b, c);
  EXPECT_EQ(triangle.CountVertices(), 3);
}
TEST_F(Triangle2dTest, geometry) {
  auto triangle = Triangle(a, b, c);
  EXPECT_EQ(triangle.Measure(), 0.5);
  auto center = triangle.Center();
  EXPECT_EQ(center.X() * 3, a.X() + b.X() + c.X());
  EXPECT_EQ(center.Y() * 3, a.Y() + b.Y() + c.Y());
  EXPECT_EQ(center.Z() * 3, a.Z() + b.Z() + c.Z());
}

class Triangle3dTest : public ::testing::Test {
 protected:
  using Real = double;
  using Triangle = Triangle<Real, 3>;
  using Point = Triangle::Point;
  Point a{0.0, 0.0, 0.0}, b{1.0, 0.0, 0.0}, c{0.0, 1.0, 0.0};
};
TEST_F(Triangle3dTest, Constructor) {
  // Test Triangle(Point const&, Point const&, Point const&):
  auto triangle = Triangle(a, b, c);
  EXPECT_EQ(triangle.CountVertices(), 3);
}
TEST_F(Triangle3dTest, geometry) {
  auto triangle = Triangle(a, b, c);
  EXPECT_EQ(triangle.Measure(), 0.5);
  auto center = triangle.Center();
  EXPECT_EQ(center.X() * 3, a.X() + b.X() + c.X());
  EXPECT_EQ(center.Y() * 3, a.Y() + b.Y() + c.Y());
  EXPECT_EQ(center.Z() * 3, a.Z() + b.Z() + c.Z());
}

}  // namespace geometry
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
