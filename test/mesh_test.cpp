// Copyright 2019 Weicheng Pei

#include "mesh.hpp"

#include <vector>

#include "gtest/gtest.h"

using pvc::cfd::Coordinate;
using pvc::cfd::Tag;
using pvc::cfd::Node;
using pvc::cfd::Edge;
using pvc::cfd::Cell;
using pvc::cfd::Mesh;

class NodeTest : public ::testing::Test {
};
TEST_F(NodeTest, Constructor) {
  auto tag = Tag{0};
  auto x = Coordinate{1.0};
  auto y = Coordinate{2.0};
  auto node = Node(tag, x, y);
  EXPECT_EQ(node.Tag(), tag);
  EXPECT_EQ(node.X(), x);
  EXPECT_EQ(node.Y(), y);
}

class EdgeTest : public ::testing::Test {
};
TEST_F(EdgeTest, Constructor) {
  auto tag = Tag{0};
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
  auto tag = Tag{0};
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
  auto mesh = pvc::cfd::Mesh();
  EXPECT_EQ(mesh.CountNodes(), 0);
  EXPECT_EQ(mesh.CountEdges(), 0);
  EXPECT_EQ(mesh.CountCells(), 0);
}
TEST_F(MeshTest, EmplaceNode) {
  auto mesh = pvc::cfd::Mesh();
  mesh.EmplaceNode(0, 0.0, 0.0);
  EXPECT_EQ(mesh.CountNodes(), 1);
}
TEST_F(MeshTest, ForEachNode) {
  auto mesh = Mesh();
  // Emplace 4 nodes:
  auto x = std::vector<Coordinate>{0.0, 1.0, 1.0, 0.0};
  auto y = std::vector<Coordinate>{0.0, 0.0, 1.0, 1.0};
  for (auto tag = 0; tag != x.size(); ++tag) {
    mesh.EmplaceNode(tag, x[tag], y[tag]);
  }
  EXPECT_EQ(mesh.CountNodes(), x.size());
  // Check each node's tag and coordinates:
  auto check_coordinates = [&x, &y](Node const& node) {
    auto tag = node.Tag();
    EXPECT_EQ(node.X(), x[tag]);
    EXPECT_EQ(node.Y(), y[tag]);
  };
  mesh.ForEachNode(check_coordinates);
}
TEST_F(MeshTest, EmplaceEdge) {
  auto mesh = pvc::cfd::Mesh();
  mesh.EmplaceNode(0, 0.0, 0.0);
  mesh.EmplaceNode(1, 1.0, 0.0);
  EXPECT_EQ(mesh.CountNodes(), 2);
  mesh.EmplaceEdge(0, 0, 1);
  EXPECT_EQ(mesh.CountEdges(), 1);
}
TEST_F(MeshTest, ForEachEdge) {
  auto mesh = Mesh();
  // Emplace 4 nodes:
  auto x = std::vector<Coordinate>{0.0, 1.0, 1.0, 0.0};
  auto y = std::vector<Coordinate>{0.0, 0.0, 1.0, 1.0};
  for (auto tag = 0; tag != x.size(); ++tag) {
    mesh.EmplaceNode(tag, x[tag], y[tag]);
  }
  EXPECT_EQ(mesh.CountNodes(), x.size());
  // Emplace 6 edges:
  auto e = 0;
  mesh.EmplaceEdge(e++, 0, 1);
  mesh.EmplaceEdge(e++, 1, 2);
  mesh.EmplaceEdge(e++, 2, 3);
  mesh.EmplaceEdge(e++, 3, 0);
  mesh.EmplaceEdge(e++, 2, 0);
  mesh.EmplaceEdge(e++, 3, 1);
  EXPECT_EQ(mesh.CountEdges(), e);
  // For each edge: head's tag < tail's tag
  auto check_edges = [](Edge const& edge) {
    EXPECT_LT(edge.Head()->Tag(), edge.Tail()->Tag());
  };
  mesh.ForEachEdge(check_edges);
}
TEST_F(MeshTest, EmplaceCell) {
  auto mesh = Mesh();
  // Emplace 4 nodes:
  auto x = std::vector<Coordinate>{0.0, 1.0, 1.0, 0.0};
  auto y = std::vector<Coordinate>{0.0, 0.0, 1.0, 1.0};
  for (auto tag = 0; tag != x.size(); ++tag) {
    mesh.EmplaceNode(tag, x[tag], y[tag]);
  }
  EXPECT_EQ(mesh.CountNodes(), x.size());
  // Emplace 2 triangular cells:
  mesh.EmplaceCell(0, {0, 1, 2});
  mesh.EmplaceCell(1, {0, 2, 3});
  EXPECT_EQ(mesh.CountCells(), 2);
  EXPECT_EQ(mesh.CountEdges(), 5);
}
TEST_F(MeshTest, ForEachCell) {
}
TEST_F(MeshTest, PositiveSide) {
  auto mesh = Mesh();
  // Emplace 4 nodes:
  auto x = std::vector<Coordinate>{0.0, 1.0, 1.0, 0.0};
  auto y = std::vector<Coordinate>{0.0, 0.0, 1.0, 1.0};
  for (auto tag = 0; tag != x.size(); ++tag) {
    mesh.EmplaceNode(tag, x[tag], y[tag]);
  }
  EXPECT_EQ(mesh.CountNodes(), x.size());
  // Emplace 5 edges:
  auto edges = std::vector<Edge*>();
  edges.emplace_back(mesh.EmplaceEdge(0, 1));
  edges.emplace_back(mesh.EmplaceEdge(1, 2));
  edges.emplace_back(mesh.EmplaceEdge(2, 3));
  edges.emplace_back(mesh.EmplaceEdge(3, 0));
  edges.emplace_back(mesh.EmplaceEdge(0, 2));
  EXPECT_EQ(mesh.CountEdges(), edges.size());
  // Emplace 2 triangular cells:
  auto cells = std::vector<Cell*>();
  cells.emplace_back(mesh.EmplaceCell(0, {0, 1, 2}));
  cells.emplace_back(mesh.EmplaceCell(1, {0, 2, 3}));
  EXPECT_EQ(mesh.CountCells(), 2);
  // Check each edge's positive side and negative side:
  // edges[0] == {nodes[0], nodes[1]}
  EXPECT_EQ(edges[0]->PositiveSide(), cells[0]);
  EXPECT_EQ(edges[0]->NegativeSide(), nullptr);
  // edges[1] == {nodes[1], nodes[2]}
  EXPECT_EQ(edges[1]->PositiveSide(), cells[0]);
  EXPECT_EQ(edges[1]->NegativeSide(), nullptr);
  // edges[2] == {nodes[0], nodes[2]}
  EXPECT_EQ(edges[2]->PositiveSide(), cells[1]);
  EXPECT_EQ(edges[2]->NegativeSide(), cells[0]);
  // edges[3] == {nodes[2], nodes[3]}
  EXPECT_EQ(edges[3]->PositiveSide(), cells[1]);
  EXPECT_EQ(edges[3]->NegativeSide(), nullptr);
  // edges[4] == {nodes[0], nodes[3]}
  EXPECT_EQ(edges[4]->PositiveSide(), nullptr);
  EXPECT_EQ(edges[4]->NegativeSide(), cells[1]);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
