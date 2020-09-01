// Copyright 2019 Weicheng Pei and Minghao Yang

#include "mini/geometry/line.hpp"

#include <vector>

#include "gtest/gtest.h"

namespace mini {
namespace geometry {

class TestLine2d : public ::testing::Test {
 protected:
  using Real = double;
  using L2 = Line<Real, 2>;
  using P2 = L2::PointType;
  P2 head{0.0, 0.0}, tail{0.3, 0.4};
};
TEST_F(TestLine2d, Constructor) {
  // Test L2(const P2 &, const P2 &):
  auto line = L2(head, tail);
  EXPECT_EQ(line.Head(), head);
  EXPECT_EQ(line.Tail(), tail);
}
TEST_F(TestLine2d, GeometricMethods) {
  auto line = L2(head, tail);
  EXPECT_EQ(line.Measure(), 0.5);
  auto c = line.Center();
  EXPECT_EQ(c.X() + c.X(), head.X() + tail.X());
  EXPECT_EQ(c.Y() + c.Y(), head.Y() + tail.Y());
  EXPECT_EQ(c.Z() + c.Z(), head.Z() + tail.Z());
}

class TestLine3d : public ::testing::Test {
 protected:
  using Real = double;
  using L3 = Line<Real, 3>;
  using P3 = L3::PointType;
  P3 head{0.0, 0.0, 0.0}, tail{0.3, 0.4, 0.0};
};
TEST_F(TestLine3d, Constructor) {
  // Test L3(const P3 &, const P3 &):
  auto line = L3(head, tail);
  EXPECT_EQ(line.Head(), head);
  EXPECT_EQ(line.Tail(), tail);
}
TEST_F(TestLine3d, GeometricMethods) {
  auto line = L3(head, tail);
  EXPECT_EQ(line.Measure(), 0.5);
  auto c = line.Center();
  EXPECT_EQ(c.X() + c.X(), head.X() + tail.X());
  EXPECT_EQ(c.Y() + c.Y(), head.Y() + tail.Y());
  EXPECT_EQ(c.Z() + c.Z(), head.Z() + tail.Z());
}

}  // namespace geometry
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
