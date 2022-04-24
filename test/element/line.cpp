// Copyright 2019 PEI Weicheng and YANG Minghao
#include "mini/element/line.hpp"

#include <vector>

#include "gtest/gtest.h"

namespace mini {
namespace element {

class LineTest : public ::testing::Test {
 protected:
  using Real = double;
  using L2 = Line<Real, 2>;
  using P2 = L2::PointType;
  const L2::IdType i{8};
  P2 head{1, {0.3, 0.0}}, tail{2, {0.0, 0.4}};
};
TEST_F(LineTest, Constructor) {
  // Test Line(Id, const Point &, const Point &):
  auto line = L2(i, head, tail);
  EXPECT_EQ(line.I(), i);
  EXPECT_EQ(line.Head(), head);
  EXPECT_EQ(line.Tail(), tail);
  EXPECT_EQ(line.Head().I(), head.I());
  EXPECT_EQ(line.Tail().I(), tail.I());
}
TEST_F(LineTest, GeometryMethods) {
  auto line = L2(i, head, tail);
  EXPECT_EQ(line.Measure(), 0.5);
  auto center = line.Center();
  EXPECT_EQ(center.X() * 2, head.X() + tail.X());
  EXPECT_EQ(center.Y() * 2, head.Y() + tail.Y());
  EXPECT_EQ(center.Z() * 2, head.Z() + tail.Z());
}
TEST_F(LineTest, ElementMethods) {
  auto line = L2(i, head, tail);
  auto integrand = [](const auto& point) { return 3.14; };
  EXPECT_EQ(line.Integrate(integrand), line.Measure() * 3.14);
}

}  // namespace element
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
