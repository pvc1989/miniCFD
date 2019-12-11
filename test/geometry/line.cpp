// Copyright 2019 Weicheng Pei and Minghao Yang
#include "mini/geometry/line.hpp"

#include <vector>

#include "gtest/gtest.h"

namespace mini {
namespace geometry {

class Line2Test : public ::testing::Test {
 protected:
  using Real = double;
  using Line = Line<Real, 2>;
  using Point = Line::Point;
  Point head{0.0, 0.0}, tail{0.3, 0.4};
};
TEST_F(Line2Test, Constructor) {
  // Test Line(Point*, Point*):
  auto line = Line(&head, &tail);
  EXPECT_EQ(line.Head(), &head);
  EXPECT_EQ(line.Tail(), &tail);
}
TEST_F(Line2Test, geometry) {
  auto line = Line(&head, &tail);
  EXPECT_EQ(line.Measure(), 0.5);
  auto c = line.Center();
  EXPECT_EQ(c.X() + c.X(), head.X() + tail.X());
  EXPECT_EQ(c.Y() + c.Y(), head.Y() + tail.Y());
}

class Line3Test : public ::testing::Test {
 protected:
  using Real = double;
  using Line = Line<Real, 3>;
  using Point = Line::Point;
  Point head{0.0, 0.0, 0.0}, tail{0.3, 0.4, 0.0};
};
TEST_F(Line3Test, Constructor) {
  // Test Line(Point*, Point*):
  auto line = Line(&head, &tail);
  EXPECT_EQ(line.Head(), &head);
  EXPECT_EQ(line.Tail(), &tail);
}
TEST_F(Line3Test, geometry) {
  auto line = Line(&head, &tail);
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
