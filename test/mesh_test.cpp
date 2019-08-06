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
}

class CellTest : public ::testing::Test {
};
TEST_F(CellTest, Constructor) {
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
