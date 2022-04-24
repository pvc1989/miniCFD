// Copyright 2019 PEI Weicheng and YANG Minghao
#include "mini/element/rectangle.hpp"

#include <vector>

#include "gtest/gtest.h"

namespace mini {
namespace element {

class RectangleTest : public ::testing::Test {
 protected:
  using Real = double;
  using Surface = Rectangle<Real, 3>;
  using Point = Surface::PointType;
  const Surface::IdType i{8};
  Point a{1, {0.0, 0.0, 0.0}}, b{2, {1.0, 0.0, 0.0}};
  Point c{3, {1.0, 1.0, 0.0}}, d{4, {0.0, 1.0, 0.0}};
};
TEST_F(RectangleTest, Constructor) {
    // Test Rectangle(Id, const Point &, const Point &, const Point &):
    auto surface = Surface(i, a, b, c, d);
    EXPECT_EQ(surface.I(), i);
    EXPECT_EQ(surface.A(), a);
    EXPECT_EQ(surface.B(), b);
    EXPECT_EQ(surface.C(), c);
    EXPECT_EQ(surface.D(), d);
}
TEST_F(RectangleTest, GeometryMethods) {
  auto surface = Surface(i, a, b, c, d);
  EXPECT_EQ(surface.Measure(), 1.0);
  auto center = surface.Center();
  EXPECT_EQ(center.X() * 4, a.X() + b.X() + c.X() + d.X());
  EXPECT_EQ(center.Y() * 4, a.Y() + b.Y() + c.Y() + d.Y());
  EXPECT_EQ(center.Z() * 4, a.Z() + b.Z() + c.Z() + d.Z());
}
TEST_F(RectangleTest, MeshMethods) {
  auto surface = Surface(i, a, b, c, d);
  auto integrand = [](const auto& point) { return 3.14; };
  EXPECT_EQ(surface.Integrate(integrand), surface.Measure() * 3.14);
}

}  // namespace element
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
