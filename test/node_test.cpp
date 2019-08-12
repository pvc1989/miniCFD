// Copyright 2019 Weicheng Pei and Minghao Yang

#include "mesh.hpp"

#include <vector>

#include "gtest/gtest.h"

using pvc::cfd::Real;
using pvc::cfd::Space;

class PointTest : public ::testing::Test {
 protected:
  using P1 = Space<1>::Point;
  using P2 = Space<2>::Point;
  using P3 = Space<3>::Point;
};
TEST_F(PointTest, Constructor) {
  auto x = Real{1.0};
  auto y = Real{2.0};
  // Test Space<2>::Point::Point(Real, Real):
  auto p2 = P2(x, y);
  EXPECT_EQ(p2.X(), x);
  EXPECT_EQ(p2.Y(), y);
}

class NodeTest : public ::testing::Test {
 protected:
  using N1 = Space<1>::Node;
  using N2 = Space<2>::Node;
  using N3 = Space<3>::Node;
  auto DefaultConstruct2() { return N2(0.0, 0.0); }
};
TEST_F(NodeTest, Constructor) {
  auto i = N2::Id(8);
  auto x = Real{1.0};
  auto y = Real{2.0};
  // Test Space<2>::Node::Node(Id, Real, Real):
  auto n2 = N2(i, x, y);
  EXPECT_EQ(n2.I(), i);
  EXPECT_EQ(n2.X(), x);
  EXPECT_EQ(n2.Y(), y);
  // Test Space<2>::Node::Node(Real, Real):
  n2 = N2(x, y);
  EXPECT_EQ(n2.I(), N2::DefaultId());
  EXPECT_EQ(n2.X(), x);
  EXPECT_EQ(n2.Y(), y);
}
TEST_F(NodeTest, ElementMethods) {
  auto n2 = DefaultConstruct2();
  EXPECT_EQ(n2.Dim(), 0);
  EXPECT_EQ(n2.Measure(), 0.0);
  auto p2 = n2.Center();
  EXPECT_EQ(p2.X(), n2.X());
  EXPECT_EQ(p2.Y(), n2.Y());
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
