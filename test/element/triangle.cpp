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
  using Point = Surface::PointType;
  const Surface::IndexType i{8};
  Point a{1, {0.0, 0.0, 0.0}}, b{2, {1.0, 0.0, 0.0}}, c{3, {0.0, 1.0, 0.0}};
};
TEST_F(TriangleTest, Constructor) {
    // Test Triangle(Index, const Point &, const Point &, const Point &):
    auto surface = Surface(i, a, b, c);
    EXPECT_EQ(surface.I(), i);
    EXPECT_EQ(surface.A(), a);
    EXPECT_EQ(surface.B(), b);
    EXPECT_EQ(surface.C(), c);
}
TEST_F(TriangleTest, GeometryMethods) {
  auto surface = Surface(i, a, b, c);
  EXPECT_EQ(surface.Measure(), 0.5);
  auto center = surface.Center();
  EXPECT_EQ(center.X() * 3, a.X() + b.X() + c.X());
  EXPECT_EQ(center.Y() * 3, a.Y() + b.Y() + c.Y());
  EXPECT_EQ(center.Z() * 3, a.Z() + b.Z() + c.Z());
}
TEST_F(TriangleTest, ElementMethods) {
  auto surface = Surface(i, a, b, c);
  auto integrand = [](const auto& point) { return 3.14; };
  EXPECT_EQ(surface.Integrate(integrand), surface.Measure() * 3.14);
}

}  // namespace element
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
