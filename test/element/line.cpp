// Copyright 2019 Weicheng Pei and Minghao Yang
#include "mini/element/line.hpp"

#include <vector>

#include "gtest/gtest.h"

namespace mini {
namespace element {

class LineTest : public ::testing::Test {
 protected:
  using Real = double;
  using Line = Line<Real, 2>;
  using Point = Line::Point;
  Point head{0.3, 0.0}, tail{0.0, 0.4};
};
TEST_F(LineTest, Constructor) {
  auto i = Line::Id{0};
  // Test Line(Id, Point*, Point*):
  auto edge = Line(i, &head, &tail);
  EXPECT_EQ(edge.I(), i);
  EXPECT_EQ(edge.Head(), &head);
  EXPECT_EQ(edge.Tail(), &tail);
  // Test Line(Point*, Point*):
  edge = Line(&head, &tail);
  EXPECT_EQ(edge.I(), Line::DefaultId());
  EXPECT_EQ(edge.Head(), &head);
  EXPECT_EQ(edge.Tail(), &tail);
}
TEST_F(LineTest, MeshMethods) {
  auto edge = Line(&head, &tail);
  EXPECT_EQ(edge.Measure(), 0.5);
  auto center = edge.Center();
  EXPECT_EQ(center.X() * 2, head.X() + tail.X());
  EXPECT_EQ(center.Y() * 2, head.Y() + tail.Y());
  auto integrand = [](const auto& point) { return 3.14; };
  EXPECT_EQ(edge.Integrate(integrand), edge.Measure() * 3.14);
}

}  // namespace element
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
