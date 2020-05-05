// Copyright 2019 Weicheng Pei and Minghao Yang

#include "mini/geometry/rectangle.hpp"

#include <vector>

#include "gtest/gtest.h"

namespace mini {
namespace geometry {

class TestRectangle : public ::testing::Test {
 protected:
  using Real = double;
  using R2 = Rectangle<Real, 2>;
  using R3 = Rectangle<Real, 3>;
  using P2 = R2::PointType;
  using P3 = R3::PointType;
  P2 a2{0.0, 0.0     }, b2{1.0, 0.0     }, c2{1.0, 1.0     }, d2{0.0, 1.0     };
  P3 a3{0.0, 0.0, 0.0}, b3{1.0, 0.0, 0.0}, c3{1.0, 1.0, 0.0}, d3{0.0, 1.0, 0.0};
};
TEST_F(TestRectangle, Constructor) {
  // Test Rectangle(const Point &, const Point &, const Point &, const Point &):
  auto r2 = R2(a2, b2, c2, d2);
  auto r3 = R3(a3, b3, c3, d3);
  EXPECT_EQ(r2.CountVertices(), 4);
  EXPECT_EQ(r3.CountVertices(), 4);
}
TEST_F(TestRectangle, GeometricMethods) {
  auto r2 = R2(a2, b2, c2, d2);
  auto r3 = R3(a3, b3, c3, d3);
  EXPECT_EQ(r2.Measure(), 1.0);
  EXPECT_EQ(r3.Measure(), 1.0);
  auto p2 = r2.Center();
  EXPECT_EQ(p2.X() * 4, a2.X() + b2.X() + c2.X() + d2.X());
  EXPECT_EQ(p2.Y() * 4, a2.Y() + b2.Y() + c2.Y() + d2.Y());
  EXPECT_EQ(p2.Z() * 4, a2.Z() + b2.Z() + c2.Z() + d2.Z());
  auto p3 = r3.Center();
  EXPECT_EQ(p3.X() * 4, a3.X() + b3.X() + c3.X() + d3.X());
  EXPECT_EQ(p3.Y() * 4, a3.Y() + b3.Y() + c3.Y() + d3.Y());
  EXPECT_EQ(p3.Z() * 4, a3.Z() + b3.Z() + c3.Z() + d3.Z());
}

}  // namespace geometry
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
