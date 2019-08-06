// Copyright 2019 Weicheng Pei

#include "mesh.hpp"

#include <vector>

#include "gtest/gtest.h"

using pvc::cfd::Coordinate;
using pvc::cfd::Node;
using pvc::cfd::Edge;
using pvc::cfd::Cell;
using pvc::cfd::Mesh;

class NodeTest : public ::testing::Test {
};
TEST_F(NodeTest, Constructor) {
  auto tag{0};
  auto x{1.0}, y{2.0};
  auto node = Node(tag, x, y);
  EXPECT_EQ(node.Tag(), tag);
  EXPECT_EQ(node.X(), x);
  EXPECT_EQ(node.Y(), y);
}

class EdgeTest : public ::testing::Test {
};
TEST_F(EdgeTest, Constructor) {
  auto tag{0};
  auto head = Node(0, 0.0, 0.0);
  auto tail = Node(1, 1.0, 0.0);
  auto edge = Edge(tag, &head, &tail);
  EXPECT_EQ(edge.Tag(), tag);
  EXPECT_EQ(edge.Head()->Tag(), head.Tag());
  EXPECT_EQ(edge.Tail()->Tag(), tail.Tag());
}

class CellTest : public ::testing::Test {
};
TEST_F(CellTest, Constructor) {
  auto tag{0};
  auto a = Node(0, 0.0, 0.0);
  auto b = Node(1, 1.0, 0.0);
  auto c = Node(2, 1.0, 1.0);
  auto ab = Edge(0, &a, &b);
  auto bc = Edge(1, &b, &c);
  auto ca = Edge(2, &c, &a);
  auto cell = Cell(tag, {&ab, &bc, &ca});
  EXPECT_EQ(cell.Tag(), tag);
}

class MeshTest : public ::testing::Test {
};
TEST_F(MeshTest, Constructor) {
}
TEST_F(MeshTest, EmplaceNode) {
}
TEST_F(MeshTest, ForEachNode) {
}
TEST_F(MeshTest, EmplaceEdge) {
}
TEST_F(MeshTest, ForEachEdge) {
}
TEST_F(MeshTest, EmplaceCell) {
}
TEST_F(MeshTest, ForEachCell) {
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
