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
