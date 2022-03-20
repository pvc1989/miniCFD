// Copyright 2019 Weicheng Pei and Minghao Yang

#include <vector>

#include "mini/dataset/dim1.hpp"

#include "gtest/gtest.h"

namespace mini {
namespace mesh {

class NodeTest : public ::testing::Test {
 protected:
  using NodeType = Node<double>;
  NodeType::IdType i{0};
};
TEST_F(NodeTest, Constructor) {
  double x = 0.3;
  auto node = NodeType(i, x);
  EXPECT_EQ(node.I(), i);
  EXPECT_EQ(node.X(), x);
}

class CellTest : public ::testing::Test {
 protected:
  using CellType = Cell<double>;
  using NodeType = typename CellType::NodeType;
  CellType::IdType i{0};
  NodeType head{1, 0.3}, tail{2, 0.4};
};
TEST_F(CellTest, Constructor) {
  auto cell = CellType(i, head, tail);
  EXPECT_EQ(cell.I(), i);
  EXPECT_EQ(cell.Head(), head);
  EXPECT_EQ(cell.Tail(), tail);
  EXPECT_EQ(cell.Head().X(), 0.3);
  EXPECT_EQ(cell.Tail().X(), 0.4);
}
TEST_F(CellTest, ElementMethods) {
  auto cell = CellType(0, head, tail);
  EXPECT_DOUBLE_EQ(cell.Measure(), 0.1);
  auto center = cell.Center();
  EXPECT_EQ(center.X() * 2, head.X() + tail.X());
  auto integrand = [](const auto& point) { return 0.618; };
  EXPECT_DOUBLE_EQ(cell.Integrate(integrand), 0.618 * cell.Measure());
}

class MeshTest : public ::testing::Test {
 protected:
  using MeshType = Mesh<double>;
  using CellType = typename MeshType::CellType;
  using NodeType = typename MeshType::NodeType;
  MeshType mesh{};
  const std::vector<double> x{0.0, 1.0, 2.0, 3.0};
};
TEST_F(MeshTest, DefaultConstructor) {
  EXPECT_EQ(mesh.CountNodes(), 0);
  EXPECT_EQ(mesh.CountCells(), 0);
}
TEST_F(MeshTest, EmplaceNode) {
  mesh.EmplaceNode(0, 0.0);
  EXPECT_EQ(mesh.CountNodes(), 1);
}
TEST_F(MeshTest, ForEachNode) {
  // Emplace 4 nodes:
  for (auto i = 0; i != x.size(); ++i) {
    mesh.EmplaceNode(i, x[i]);
  }
  EXPECT_EQ(mesh.CountNodes(), x.size());
  // Check each NodeType's index and coordinates:
  mesh.ForEachNode([&](NodeType const& node) {
    auto i = node.I();
    EXPECT_EQ(node.X(), x[i]);
  });
}
TEST_F(MeshTest, EmplaceCell) {
  mesh.EmplaceNode(0, 0.0);
  mesh.EmplaceNode(1, 1.0);
  EXPECT_EQ(mesh.CountNodes(), 2);
  mesh.EmplaceCell(0, 0, 1);
  EXPECT_EQ(mesh.CountCells(), 1);
}
TEST_F(MeshTest, ForEachCell) {
  /*
     0 ----- 1 ----- 2 ----- 3
  */
  // Emplace 4 nodes:
  for (auto i = 0; i != x.size(); ++i) {
    mesh.EmplaceNode(i, x[i]);
  }
  EXPECT_EQ(mesh.CountNodes(), x.size());
  // Emplace 3 cells:
  auto e = 0;
  mesh.EmplaceCell(e++, 0, 1);
  mesh.EmplaceCell(e++, 1, 2);
  mesh.EmplaceCell(e++, 2, 3);
  EXPECT_EQ(mesh.CountCells(), e);
  // For each cell: head's index < tail's index
  mesh.ForEachCell([](CellType const& cell) {
    EXPECT_LT(cell.Head().I(), cell.Tail().I());
  });
}
TEST_F(MeshTest, GetSide) {
  /*
     0 ----- 1 ----- 2 ----- 3
  */
  // Emplace 4 nodes:
  auto nodes = std::vector<NodeType*>();
  for (auto i = 0; i != x.size(); ++i) {
    nodes.emplace_back(mesh.EmplaceNode(i, x[i]));
  }
  EXPECT_EQ(mesh.CountNodes(), x.size());
  // Emplace 3 line cells:
  auto cells = std::vector<CellType*>();
  cells.emplace_back(mesh.EmplaceCell(0, 0, 1));
  cells.emplace_back(mesh.EmplaceCell(1, 1, 2));
  cells.emplace_back(mesh.EmplaceCell(2, 2, 3));
  EXPECT_EQ(mesh.CountCells(), cells.size());
  // Check each node's positive side and negative side:
  // nodes[0]
  EXPECT_EQ(nodes[0]->GetPositiveSide(), nullptr);
  EXPECT_EQ(nodes[0]->GetNegativeSide(), cells[0]);
  // nodes[1]
  EXPECT_EQ(nodes[1]->GetPositiveSide(), cells[0]);
  EXPECT_EQ(nodes[1]->GetNegativeSide(), cells[1]);
  // nodes[2]
  EXPECT_EQ(nodes[2]->GetPositiveSide(), cells[1]);
  EXPECT_EQ(nodes[2]->GetNegativeSide(), cells[2]);
  // nodes[3]
  EXPECT_EQ(nodes[3]->GetPositiveSide(), cells[2]);
  EXPECT_EQ(nodes[3]->GetNegativeSide(), nullptr);
}

}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
