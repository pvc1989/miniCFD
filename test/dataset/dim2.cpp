// Copyright 2019 Weicheng Pei and Minghao Yang

#include <vector>

#include "mini/dataset/dim2.hpp"

#include "gtest/gtest.h"

namespace mini {
namespace mesh {

class WallTest : public ::testing::Test {
 protected:
  using WallType = Wall<double>;
  using NodeType = WallType::NodeType;
  WallType::IdType i{0};
  NodeType head{1, 0.3, 0.0}, tail{2, 0.0, 0.4};
};
TEST_F(WallTest, Constructor) {
  auto wall = WallType(i, head, tail);
  EXPECT_EQ(wall.I(), i);
  EXPECT_EQ(wall.Head(), head);
  EXPECT_EQ(wall.Tail(), tail);
}
TEST_F(WallTest, GeometryMethods) {
  auto wall = WallType(i, head, tail);
  EXPECT_DOUBLE_EQ(wall.Measure(), 0.5);
  auto center = wall.Center();
  EXPECT_EQ(center.X() * 2, head.X() + tail.X());
  EXPECT_EQ(center.Y() * 2, head.Y() + tail.Y());
  EXPECT_EQ(center.Z() * 2, head.Z() + tail.Z());
}
TEST_F(WallTest, ElementMethods) {
  auto wall = WallType(i, head, tail);
  auto integrand = [](const auto& point) { return 0.618; };
  EXPECT_DOUBLE_EQ(wall.Integrate(integrand), 0.618 * wall.Measure());
}

class TriangleTest : public ::testing::Test {
 protected:
  using CellType = Triangle<double>;
  using WallType = CellType::WallType;
  using NodeType = WallType::NodeType;
  CellType::IdType i{0};
  NodeType a{1, 0.0, 0.0}, b{2, 1.0, 0.0}, c{3, 0.0, 1.0};
  WallType ab{1, a, b}, bc{2, b, c}, ca{3, c, a};
};
TEST_F(TriangleTest, Constructor) {
  auto triangle = CellType(i, a, b, c, {&ab, &bc, &ca});
  EXPECT_EQ(triangle.I(), i);
}
TEST_F(TriangleTest, GeometryMethods) {
  auto cell = CellType(i, a, b, c, {&ab, &bc, &ca});
  EXPECT_DOUBLE_EQ(cell.Measure(), 0.5);
  auto center = cell.Center();
  EXPECT_EQ(center.X() * 3, a.X() + b.X() + c.X());
  EXPECT_EQ(center.Y() * 3, a.Y() + b.Y() + c.Y());
  EXPECT_EQ(center.Z() * 3, a.Z() + b.Z() + c.Z());
}
TEST_F(TriangleTest, ElementMethods) {
  auto cell = CellType(i, a, b, c, {&ab, &bc, &ca});
  auto integrand = [](const auto& point) { return 0.618; };
  EXPECT_DOUBLE_EQ(cell.Integrate(integrand), 0.618 * cell.Measure());
}

class RectangleTest : public ::testing::Test {
 protected:
  using CellType = Rectangle<double>;
  using WallType = CellType::WallType;
  using NodeType = WallType::NodeType;
  CellType::IdType i{0};
  NodeType a{1, 0.0, 0.0}, b{2, 1.0, 0.0}, c{3, 1.0, 1.0}, d{4, 0.0, 1.0};
  WallType ab{1, a, b}, bc{2, b, c}, cd{3, c, d}, da{4, d, a};
};
TEST_F(RectangleTest, Constructor) {
  auto rectangle = CellType(i, a, b, c, d, {&ab, &bc, &cd, &da});
  EXPECT_EQ(rectangle.I(), i);
}
TEST_F(RectangleTest, GeometryMethods) {
  auto cell = CellType(i, a, b, c, d, {&ab, &bc, &cd, &da});
  EXPECT_DOUBLE_EQ(cell.Measure(), 1.0);
  auto center = cell.Center();
  EXPECT_EQ(center.X() * 4, a.X() + b.X() + c.X() + d.X());
  EXPECT_EQ(center.Y() * 4, a.Y() + b.Y() + c.Y() + d.Y());
  EXPECT_EQ(center.Z() * 4, a.Z() + b.Z() + c.Z() + d.Z());
}
TEST_F(RectangleTest, ElementMethods) {
  auto cell = CellType(i, a, b, c, d, {&ab, &bc, &cd, &da});
  auto integrand = [](const auto& point) { return 0.618; };
  EXPECT_DOUBLE_EQ(cell.Integrate(integrand), 0.618 * cell.Measure());
}

class MeshTest : public ::testing::Test {
 protected:
  using MeshType = Mesh<double>;
  using CellType = MeshType::CellType;
  using WallType = MeshType::WallType;
  using NodeType = MeshType::NodeType;
  MeshType mesh{};
  const std::vector<double> x{0.0, 1.0, 1.0, 0.0}, y{0.0, 0.0, 1.0, 1.0};
};
TEST_F(MeshTest, DefaultConstructor) {
  EXPECT_EQ(mesh.CountNodes(), 0);
  EXPECT_EQ(mesh.CountWalls(), 0);
  EXPECT_EQ(mesh.CountCells(), 0);
}
TEST_F(MeshTest, EmplaceNode) {
  EXPECT_EQ(mesh.CountNodes(), 0);
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
  mesh.ForEachNode([&](NodeType const& node) {
    auto i = node.I();
    EXPECT_EQ(node.X(), x[i]);
    EXPECT_EQ(node.Y(), y[i]);
  });
}
TEST_F(MeshTest, EmplaceWall) {
  EXPECT_EQ(mesh.CountNodes(), 0);
  mesh.EmplaceNode(0, 0.0, 0.0);
  mesh.EmplaceNode(1, 1.0, 0.0);
  EXPECT_EQ(mesh.CountNodes(), 2);
  EXPECT_EQ(mesh.CountWalls(), 0);
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
  for (auto n = 0; n != x.size(); ++n) {
    mesh.EmplaceNode(n, x[n], y[n]);
  }
  EXPECT_EQ(mesh.CountNodes(), x.size());
  // Emplace 6 walls:
  auto w = 0;
  mesh.EmplaceWall(w++, 0, 1);
  mesh.EmplaceWall(w++, 1, 2);
  mesh.EmplaceWall(w++, 2, 3);
  mesh.EmplaceWall(w++, 3, 0);
  mesh.EmplaceWall(w++, 2, 0);
  mesh.EmplaceWall(w++, 3, 1);
  EXPECT_EQ(mesh.CountWalls(), w);
  // For each wall: head's index < tail's index
  mesh.ForEachWall([](WallType const& wall) {
    EXPECT_LT(wall.Head().I(), wall.Tail().I());
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
  for (auto n = 0; n != x.size(); ++n) {
    mesh.EmplaceNode(n, x[n], y[n]);
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
  for (auto n = 0; n != x.size(); ++n) {
    mesh.EmplaceNode(n, x[n], y[n]);
  }
  // Emplace 1 clock-wise triangle and 1 clock-wise rectangle:
  mesh.EmplaceCell(0, {0, 2, 1});
  mesh.EmplaceCell(2, {0, 3, 2, 1});
  EXPECT_EQ(mesh.CountCells(), 2);
  EXPECT_EQ(mesh.CountWalls(), 5);
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
  for (auto n = 0; n != x.size(); ++n) {
    mesh.EmplaceNode(n, x[n], y[n]);
  }
  EXPECT_EQ(mesh.CountNodes(), x.size());
  // Emplace 5 walls:
  auto walls = std::vector<WallType*>();
  walls.emplace_back(mesh.EmplaceWall(0, 1));
  walls.emplace_back(mesh.EmplaceWall(1, 2));
  walls.emplace_back(mesh.EmplaceWall(2, 3));
  walls.emplace_back(mesh.EmplaceWall(3, 0));
  walls.emplace_back(mesh.EmplaceWall(0, 2));
  EXPECT_EQ(mesh.CountWalls(), walls.size());
  // Emplace 2 triangular cells:
  auto cells = std::vector<CellType*>();
  cells.emplace_back(mesh.EmplaceCell(0, {0, 1, 2}));
  cells.emplace_back(mesh.EmplaceCell(1, {0, 2, 3}));
  EXPECT_EQ(mesh.CountCells(), 2);
  EXPECT_EQ(mesh.CountWalls(), walls.size());
  // Check each wall's positive side and negative side:
  // walls[0] == {nodes[0], nodes[1]}
  EXPECT_EQ(walls[0]->GetPositiveSide(), cells[0]);
  EXPECT_EQ(walls[0]->GetNegativeSide(), nullptr);
  // walls[1] == {nodes[1], nodes[2]}
  EXPECT_EQ(walls[1]->GetPositiveSide(), cells[0]);
  EXPECT_EQ(walls[1]->GetNegativeSide(), nullptr);
  // walls[4] == {nodes[0], nodes[2]}
  EXPECT_EQ(walls[4]->GetPositiveSide(), cells[1]);
  EXPECT_EQ(walls[4]->GetNegativeSide(), cells[0]);
  // walls[2] == {nodes[2], nodes[3]}
  EXPECT_EQ(walls[2]->GetPositiveSide(), cells[1]);
  EXPECT_EQ(walls[2]->GetNegativeSide(), nullptr);
  // walls[3] == {nodes[0], nodes[3]}
  EXPECT_EQ(walls[3]->GetPositiveSide(), nullptr);
  EXPECT_EQ(walls[3]->GetNegativeSide(), cells[1]);
}

}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
