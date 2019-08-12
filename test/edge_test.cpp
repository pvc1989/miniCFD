// Copyright 2019 Weicheng Pei and Minghao Yang

#include "mesh.hpp"

#include <vector>

#include "gtest/gtest.h"

using pvc::cfd::Real;
using pvc::cfd::Space;

class Line2Test : public ::testing::Test {
 protected:
  using Point = Space<2>::Point;
  using Line = Space<2>::Line;
};
TEST_F(Line2Test, Constructor) {
  auto head = Point(0.0, 0.0);
  auto tail = Point(0.3, 0.4);
  // Test Space<2>::Line::Line(Point*, Point*):
  auto line = Line(&head, &tail);
  EXPECT_EQ(line.Head(), &head);
  EXPECT_EQ(line.Tail(), &tail);
}

class Edge2Test : public ::testing::Test {
 protected:
  using Node = Space<2>::Node;
  using Edge = Space<2>::Edge;
  // auto DefaultConstruct2() { return N2(0.0, 0.0); }
};
TEST_F(Edge2Test, Constructor) {
  auto i = Edge::Id(8);
  auto head = Node(0.0, 0.0);
  auto tail = Node(0.3, 0.4);
  // Test Space<2>::Edge::Edge(Id, Node*, Node*):
  auto edge = Edge(i, &head, &tail);
  EXPECT_EQ(edge.I(), i);
  EXPECT_EQ(edge.Head(), &head);
  EXPECT_EQ(edge.Tail(), &tail);
  // Test Space<2>::Edge::Edge(Node*, Node*):
  edge = Edge(&head, &tail);
  EXPECT_EQ(edge.I(), Edge::DefaultId());
  EXPECT_EQ(edge.Head(), &head);
  EXPECT_EQ(edge.Tail(), &tail);
}
TEST_F(Edge2Test, ElementMethods) {
  auto i = Edge::Id(8);
  auto head = Node(0.0, 0.0);
  auto tail = Node(0.3, 0.4);
  auto edge = Edge(i, &head, &tail);
  EXPECT_EQ(edge.Dim(), 1);
  EXPECT_EQ(edge.Measure(), 0.5);
  auto p2 = edge.Center();
  EXPECT_EQ(p2.X(), 0.15);
  EXPECT_EQ(p2.Y(), 0.2);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
