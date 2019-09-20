// Copyright 2019 Weicheng Pei and Minghao Yang

#include <vector>

#include "mini/mesh/dim2.hpp"

#include "gtest/gtest.h"

namespace mini {
namespace mesh {

class WallTest : public ::testing::Test {
 protected:
  using Wall = Wall<double>;
  using Node = Wall::Node;
  Wall::Id i{0};
  Node head{0.3, 0.0}, tail{0.0, 0.4};
};
TEST_F(WallTest, Constructor) {
  auto wall = Wall(i, &head, &tail);
  EXPECT_EQ(wall.I(), i);
  EXPECT_EQ(wall.Head(), &head);
  EXPECT_EQ(wall.Tail(), &tail);
}
TEST_F(WallTest, ElementMethods) {
  auto wall = Wall(&head, &tail);
  EXPECT_DOUBLE_EQ(wall.Measure(), 0.5);
  auto center = wall.Center();
  EXPECT_EQ(center.X() * 2, head.X() + tail.X());
  EXPECT_EQ(center.Y() * 2, head.Y() + tail.Y());
  auto integrand = [](const auto& point) { return 0.618; };
  EXPECT_DOUBLE_EQ(wall.Integrate(integrand), 0.618 * wall.Measure());
}

class TriangleTest : public ::testing::Test {
 protected:
  using Cell = Triangle<double>;
  using Wall = Cell::Wall;
  using Node = Wall::Node;
  Cell::Id i{0};
  Node a{0.0, 0.0}, b{1.0, 0.0}, c{0.0, 1.0};
  Wall ab{&a, &b}, bc{&b, &c}, ca{&c, &a};
};
TEST_F(TriangleTest, Constructor) {
  auto triangle = Cell(i, &a, &b, &c, {&ab, &bc, &ca});
  EXPECT_EQ(triangle.I(), i);
}
TEST_F(TriangleTest, ElementMethods) {
  auto triangle = Cell(i, &a, &b, &c, {&ab, &bc, &ca});
  EXPECT_DOUBLE_EQ(triangle.Measure(), 0.5);
  auto center = triangle.Center();
  EXPECT_EQ(center.X() * 3, a.X() + b.X() + c.X());
  EXPECT_EQ(center.Y() * 3, a.Y() + b.Y() + c.Y());
  auto integrand = [](const auto& point) { return 0.618; };
  EXPECT_DOUBLE_EQ(triangle.Integrate(integrand), 0.618 * triangle.Measure());
}

class RectangleTest : public ::testing::Test {
 protected:
  using Cell = Rectangle<double>;
  using Wall = Cell::Wall;
  using Node = Wall::Node;
  Cell::Id i{0};
  Node a{0.0, 0.0}, b{1.0, 0.0}, c{1.0, 1.0}, d{0.0, 1.0};
  Wall ab{&a, &b}, bc{&b, &c}, cd{&c, &d}, da{&d, &a};
};
TEST_F(RectangleTest, Constructor) {
  auto rectangle = Cell(i, &a, &b, &c, &d, {&ab, &bc, &cd, &da});
  EXPECT_EQ(rectangle.I(), i);
}
TEST_F(RectangleTest, ElementMethods) {
  auto rectangle = Cell(i, &a, &b, &c, &d, {&ab, &bc, &cd, &da});
  EXPECT_DOUBLE_EQ(rectangle.Measure(), 1.0);
  auto center = rectangle.Center();
  EXPECT_EQ(center.X() * 4, a.X() + b.X() + c.X() + d.X());
  EXPECT_EQ(center.Y() * 4, a.Y() + b.Y() + c.Y() + d.Y());
  auto integrand = [](const auto& point) { return 0.618; };
  EXPECT_DOUBLE_EQ(rectangle.Integrate(integrand), 0.618 * rectangle.Measure());
}

class MeshTest : public ::testing::Test {
 protected:
  using Mesh = Mesh<double>;
  using Cell = Mesh::Cell;
  using Wall = Mesh::Wall;
  using Node = Mesh::Node;
  Mesh mesh{};
  const std::vector<double> x{0.0, 1.0, 1.0, 0.0}, y{0.0, 0.0, 1.0, 1.0};
};
TEST_F(MeshTest, DefaultConstructor) {
  EXPECT_EQ(mesh.CountNodes(), 0);
  EXPECT_EQ(mesh.CountWalls(), 0);
  EXPECT_EQ(mesh.CountCells(), 0);
}
TEST_F(MeshTest, EmplaceNode) {
  mesh.EmplaceNode(0, 0.0, 0.0);
  EXPECT_EQ(mesh.CountNodes(), 1);
}
TEST_F(MeshTest, ForEachNode) {
  // Emplace 4 nodes:
  for (auto i = 0; i != x.size(); ++i) {
    mesh.EmplaceNode(i, x[i], y[i]);
  }
  EXPECT_EQ(mesh.CountNodes(), x.size());
  // Check each node's index and coordinates:
  mesh.ForEachNode([&](Node const& node) {
    auto i = node.I();
    EXPECT_EQ(node.X(), x[i]);
    EXPECT_EQ(node.Y(), y[i]);
  });
}
TEST_F(MeshTest, EmplaceWall) {
  mesh.EmplaceNode(0, 0.0, 0.0);
  mesh.EmplaceNode(1, 1.0, 0.0);
  EXPECT_EQ(mesh.CountNodes(), 2);
  mesh.EmplaceWall(0, 0, 1);
  EXPECT_EQ(mesh.CountWalls(), 1);
}
TEST_F(MeshTest, ForEachWall) {
  /*
     3 ----- 2
     | \   / |
     |   X   |
     | /   \ |
     0 ----- 1
  */
  // Emplace 4 nodes:
  for (auto i = 0; i != x.size(); ++i) {
    mesh.EmplaceNode(i, x[i], y[i]);
  }
  EXPECT_EQ(mesh.CountNodes(), x.size());
  // Emplace 6 walls:
  auto e = 0;
  mesh.EmplaceWall(e++, 0, 1);
  mesh.EmplaceWall(e++, 1, 2);
  mesh.EmplaceWall(e++, 2, 3);
  mesh.EmplaceWall(e++, 3, 0);
  mesh.EmplaceWall(e++, 2, 0);
  mesh.EmplaceWall(e++, 3, 1);
  EXPECT_EQ(mesh.CountWalls(), e);
  // For each wall: head's index < tail's index
  mesh.ForEachWall([](Wall const& wall) {
    EXPECT_LT(wall.Head()->I(), wall.Tail()->I());
  });
}
TEST_F(MeshTest, EmplaceCell) {
  /*
     3 ----- 2
     | (0) / |
     |   /   |
     | / (1) |
     0 ----- 1
  */
  // Emplace 4 nodes:
  for (auto i = 0; i != x.size(); ++i) {
    mesh.EmplaceNode(i, x[i], y[i]);
  }
  EXPECT_EQ(mesh.CountNodes(), x.size());
  // Emplace 2 triangular cells:
  mesh.EmplaceCell(0, {0, 1, 2});
  mesh.EmplaceCell(1, {0, 2, 3});
  EXPECT_EQ(mesh.CountCells(), 2);
  EXPECT_EQ(mesh.CountWalls(), 5);
}
TEST_F(MeshTest, ForEachCell) {
  /*
     3 ----- 2
     |     / |
     |   /   |
     | /     |
     0 ----- 1
  */
  // Emplace 4 nodes:
  for (auto i = 0; i != x.size(); ++i) {
    mesh.EmplaceNode(i, x[i], y[i]);
  }
  // Emplace 1 clock-wise triangle and 1 clock-wise rectangle:
  mesh.EmplaceCell(0, {0, 2, 1});
  mesh.EmplaceCell(2, {0, 3, 2, 1});
  // Check counter-clock-wise property:
  mesh.ForEachCell([](Cell const& cell) {
    auto a = cell.GetPoint(0);
    auto b = cell.GetPoint(1);
    auto c = cell.GetPoint(2);
    EXPECT_FALSE(a->IsClockWise(b, c));
  });
}
TEST_F(MeshTest, GetSide) {
  /*
     3 -- [2] -- 2
     |  (1)   /  |
    [3]   [4]   [1]
     |  /   (0)  |
     0 -- [0] -- 1
  */
  // Emplace 4 nodes:
  for (auto i = 0; i != x.size(); ++i) {
    mesh.EmplaceNode(i, x[i], y[i]);
  }
  EXPECT_EQ(mesh.CountNodes(), x.size());
  // Emplace 5 walls:
  auto walls = std::vector<Wall*>();
  walls.emplace_back(mesh.EmplaceWall(0, 1));
  walls.emplace_back(mesh.EmplaceWall(1, 2));
  walls.emplace_back(mesh.EmplaceWall(2, 3));
  walls.emplace_back(mesh.EmplaceWall(3, 0));
  walls.emplace_back(mesh.EmplaceWall(0, 2));
  EXPECT_EQ(mesh.CountWalls(), walls.size());
  // Emplace 2 triangular cells:
  auto cells = std::vector<Cell*>();
  cells.emplace_back(mesh.EmplaceCell(0, {0, 1, 2}));
  cells.emplace_back(mesh.EmplaceCell(1, {0, 2, 3}));
  EXPECT_EQ(mesh.CountCells(), 2);
  EXPECT_EQ(mesh.CountWalls(), walls.size());
  // Check each wall's positive side and negative side:
  // walls[0] == {nodes[0], nodes[1]}
  EXPECT_EQ(walls[0]->GetSide<+1>(), cells[0]);
  EXPECT_EQ(walls[0]->GetSide<-1>(), nullptr);
  // walls[1] == {nodes[1], nodes[2]}
  EXPECT_EQ(walls[1]->GetSide<+1>(), cells[0]);
  EXPECT_EQ(walls[1]->GetSide<-1>(), nullptr);
  // walls[4] == {nodes[0], nodes[2]}
  EXPECT_EQ(walls[4]->GetSide<+1>(), cells[1]);
  EXPECT_EQ(walls[4]->GetSide<-1>(), cells[0]);
  // walls[2] == {nodes[2], nodes[3]}
  EXPECT_EQ(walls[2]->GetSide<+1>(), cells[1]);
  EXPECT_EQ(walls[2]->GetSide<-1>(), nullptr);
  // walls[3] == {nodes[0], nodes[3]}
  EXPECT_EQ(walls[3]->GetSide<+1>(), nullptr);
  EXPECT_EQ(walls[3]->GetSide<-1>(), cells[1]);
}

}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
