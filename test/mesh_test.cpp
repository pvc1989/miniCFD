// Copyright 2019 Weicheng Pei

#include "mesh.hpp"

#include <vector>

#include "gtest/gtest.h"

using pvc::cfd::Real;
using pvc::cfd::Point;
using pvc::cfd::Node;
using pvc::cfd::Edge;
using pvc::cfd::Cell;
using pvc::cfd::Triangle;
using pvc::cfd::Rectangle;
using pvc::cfd::Mesh;

class NodeTest : public ::testing::Test {
};
TEST_F(NodeTest, Constructor) {
  auto i = Node::Id{0};
  auto x = Real{1.0};
  auto y = Real{2.0};
  auto node = Node(i, x, y);
  EXPECT_EQ(node.I(), i);
  EXPECT_EQ(node.X(), x);
  EXPECT_EQ(node.Y(), y);
}

class EdgeTest : public ::testing::Test {
};
TEST_F(EdgeTest, Constructor) {
  auto i = Edge::Id{0};
  auto head = Node(0, 0.0, 0.0);
  auto tail = Node(1, 1.0, 0.0);
  auto edge = Edge(i, &head, &tail);
  EXPECT_EQ(edge.I(), i);
  EXPECT_EQ(edge.Head()->I(), head.I());
  EXPECT_EQ(edge.Tail()->I(), tail.I());
}
TEST_F(EdgeTest, ElementMethods) {
  Real length = 3.14;
  auto head = Node(0, 0.0, 0.0);
  auto tail = Node(1, 0.0, length);
  auto edge = Edge(1, &head, &tail);
  EXPECT_DOUBLE_EQ(edge.Measure(), length);
  auto center = edge.Center();
  EXPECT_EQ(center.X() * 2, head.X() + tail.X());
  EXPECT_EQ(center.Y() * 2, head.Y() + tail.Y());
  auto integrand = [](Point point) { return 0.618; };
  EXPECT_DOUBLE_EQ(edge.Integrate(integrand), 0.618 * length);
}

class CellTest : public ::testing::Test {
};
TEST_F(CellTest, Constructor) {
  auto i = Cell::Id{0};
  auto a = Node(0, 0.0, 0.0);
  auto b = Node(1, 1.0, 0.0);
  auto c = Node(2, 1.0, 1.0);
  auto ab = Edge(0, &a, &b);
  auto bc = Edge(1, &b, &c);
  auto ca = Edge(2, &c, &a);
  auto cell = Cell(i, {&ab, &bc, &ca});
  EXPECT_EQ(cell.I(), i);
}

class TriangleTest : public ::testing::Test {
};
TEST_F(TriangleTest, Constructor) {
  auto i = Cell::Id{0};
  auto a = Node(0, 0.0, 0.0);
  auto b = Node(1, 1.0, 0.0);
  auto c = Node(2, 0.0, 1.0);
  auto ab = Edge(0, &a, &b);
  auto bc = Edge(1, &b, &c);
  auto ca = Edge(2, &c, &a);
  auto edges = {&ab, &bc, &ca};
  auto vertices = {&a, &b, &c};
  auto triangle = Triangle(i, edges, vertices);
  EXPECT_EQ(triangle.I(), i);
}
TEST_F(TriangleTest, ElementMethods) {
  Real area = 0.5;
  auto i = Cell::Id{0};
  auto a = Node(0, 0.0, 0.0);
  auto d = Node(1, 0.5, 0.0);
  auto b = Node(2, 1.0, 0.0);
  auto c = Node(3, 0.0, 1.0);
  auto ad = Edge(0, &a, &d);
  auto db = Edge(0, &d, &b);
  auto bc = Edge(1, &b, &c);
  auto ca = Edge(2, &c, &a);
  auto edges = {&ad, &db, &bc, &ca};
  auto vertices = {&a, &b, &c};
  auto triangle = Triangle(i, edges, vertices);
  EXPECT_DOUBLE_EQ(triangle.Measure(), area);
  auto center = triangle.Center();
  EXPECT_EQ(center.X() * 3, a.X() + b.X() + c.X());
  EXPECT_EQ(center.Y() * 3, a.Y() + b.Y() + c.Y());
  auto integrand = [](Point point) { return 0.618; };
  EXPECT_DOUBLE_EQ(triangle.Integrate(integrand), 0.618 * area);
}

class RectangleTest : public ::testing::Test {
};
TEST_F(RectangleTest, Constructor) {
  auto i = Cell::Id{0};
  auto a = Node(0, 0.0, 0.0);
  auto b = Node(1, 1.0, 0.0);
  auto c = Node(2, 1.0, 1.0);
  auto d = Node(3, 0.0, 1.0);
  auto ab = Edge(0, &a, &b);
  auto bc = Edge(1, &b, &c);
  auto cd = Edge(2, &c, &d);
  auto da = Edge(3, &d, &a);
  auto edges = {&ab, &bc, &cd, &da};
  auto vertices = {&a, &b, &c, &d};
  auto rectangle = Rectangle(i, edges, vertices);
  EXPECT_EQ(rectangle.I(), i);
}
TEST_F(RectangleTest, ElementMethods) {
  Real area = 2.0;
  auto i = Cell::Id{0};
  auto a = Node(0, 0.0, 0.0);
  auto b = Node(1, 2.0, 0.0);
  auto c = Node(2, 2.0, 1.0);
  auto d = Node(3, 0.0, 1.0);
  auto ab = Edge(0, &a, &b);
  auto bc = Edge(1, &b, &c);
  auto cd = Edge(2, &c, &d);
  auto da = Edge(3, &d, &a);
  auto edges = {&ab, &bc, &cd, &da};
  auto vertices = {&a, &b, &c, &d};
  auto rectangle = Rectangle(i, edges, vertices);
  EXPECT_DOUBLE_EQ(rectangle.Measure(), area);
  auto center = rectangle.Center();
  EXPECT_EQ(center.X() * 4, a.X() + b.X() + c.X() + d.X());
  EXPECT_EQ(center.Y() * 4, a.Y() + b.Y() + c.Y() + d.Y());
  auto integrand = [](Point point) { return 0.618; };
  EXPECT_DOUBLE_EQ(rectangle.Integrate(integrand), 0.618 * area);
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
  auto x = std::vector<Real>{0.0, 1.0, 1.0, 0.0};
  auto y = std::vector<Real>{0.0, 0.0, 1.0, 1.0};
  for (auto i = 0; i != x.size(); ++i) {
    mesh.EmplaceNode(i, x[i], y[i]);
  }
  EXPECT_EQ(mesh.CountNodes(), x.size());
  // Check each node's index and coordinates:
  auto check_coordinates = [&x, &y](Node const& node) {
    auto i = node.I();
    EXPECT_EQ(node.X(), x[i]);
    EXPECT_EQ(node.Y(), y[i]);
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
  auto x = std::vector<Real>{0.0, 1.0, 1.0, 0.0};
  auto y = std::vector<Real>{0.0, 0.0, 1.0, 1.0};
  for (auto i = 0; i != x.size(); ++i) {
    mesh.EmplaceNode(i, x[i], y[i]);
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
  // For each edge: head's index < tail's index
  auto check_edges = [](Edge const& edge) {
    EXPECT_LT(edge.Head()->I(), edge.Tail()->I());
  };
  mesh.ForEachEdge(check_edges);
}
TEST_F(MeshTest, EmplaceCell) {
  auto mesh = Mesh();
  // Emplace 4 nodes:
  auto x = std::vector<Real>{0.0, 1.0, 1.0, 0.0};
  auto y = std::vector<Real>{0.0, 0.0, 1.0, 1.0};
  for (auto i = 0; i != x.size(); ++i) {
    mesh.EmplaceNode(i, x[i], y[i]);
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
  auto x = std::vector<Real>{0.0, 1.0, 1.0, 0.0};
  auto y = std::vector<Real>{0.0, 0.0, 1.0, 1.0};
  for (auto i = 0; i != x.size(); ++i) {
    mesh.EmplaceNode(i, x[i], y[i]);
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
  EXPECT_EQ(mesh.CountEdges(), edges.size());
  // Check each edge's positive side and negative side:
  // edges[0] == {nodes[0], nodes[1]}
  EXPECT_EQ(edges[0]->PositiveSide(), cells[0]);
  EXPECT_EQ(edges[0]->NegativeSide(), nullptr);
  // edges[1] == {nodes[1], nodes[2]}
  EXPECT_EQ(edges[1]->PositiveSide(), cells[0]);
  EXPECT_EQ(edges[1]->NegativeSide(), nullptr);
  // edges[4] == {nodes[0], nodes[2]}
  EXPECT_EQ(edges[4]->PositiveSide(), cells[1]);
  EXPECT_EQ(edges[4]->NegativeSide(), cells[0]);
  // edges[2] == {nodes[2], nodes[3]}
  EXPECT_EQ(edges[2]->PositiveSide(), cells[1]);
  EXPECT_EQ(edges[2]->NegativeSide(), nullptr);
  // edges[3] == {nodes[0], nodes[3]}
  EXPECT_EQ(edges[3]->PositiveSide(), nullptr);
  EXPECT_EQ(edges[3]->NegativeSide(), cells[1]);
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
