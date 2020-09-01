// Copyright 2019 Weicheng Pei and Minghao Yang

#include "mini/geometry/triangle.hpp"

#include <vector>

#include "gtest/gtest.h"

namespace mini {
namespace geometry {

class TestTriangle : public ::testing::Test {
 protected:
  using Real = double;
  using T2 = Triangle<Real, 2>;
  using T3 = Triangle<Real, 3>;
  using P2 = T2::PointType;
  using P3 = T3::PointType;
  P2 a2{0.0, 0.0     }, b2{1.0, 0.0     }, c2{0.0, 1.0     };
  P3 a3{0.0, 0.0, 0.0}, b3{1.0, 0.0, 0.0}, c3{0.0, 1.0, 0.0};
};
TEST_F(TestTriangle, Constructor) {
  // Test Triangle(const Point &, const Point &, const Point &):
  auto t2 = T2(a2, b2, c2);
  auto t3 = T3(a3, b3, c3);
  EXPECT_EQ(t2.CountVertices(), 3);
  EXPECT_EQ(t3.CountVertices(), 3);
}
TEST_F(TestTriangle, GeometricMethods) {
  auto t2 = T2(a2, b2, c2);
  auto t3 = T3(a3, b3, c3);
  EXPECT_EQ(t2.Measure(), 0.5);
  EXPECT_EQ(t3.Measure(), 0.5);
  auto p2 = t2.Center();
  EXPECT_EQ(p2.X() * 3, a2.X() + b2.X() + c2.X());
  EXPECT_EQ(p2.Y() * 3, a2.Y() + b2.Y() + c2.Y());
  EXPECT_EQ(p2.Z() * 3, a2.Z() + b2.Z() + c2.Z());
  auto p3 = t3.Center();
  EXPECT_EQ(p3.X() * 3, a3.X() + b3.X() + c3.X());
  EXPECT_EQ(p3.Y() * 3, a3.Y() + b3.Y() + c3.Y());
  EXPECT_EQ(p3.Z() * 3, a3.Z() + b3.Z() + c3.Z());
}

}  // namespace geometry
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
