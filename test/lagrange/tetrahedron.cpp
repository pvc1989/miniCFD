//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <vector>

#include "mini/lagrange/cell.hpp"
#include "mini/lagrange/tetrahedron.hpp"

#include "gtest/gtest.h"

class TestLagrangeTetrahedron4 : public ::testing::Test {
 protected:
  static constexpr int kPoints = 24;
  using Lagrange = mini::lagrange::Tetrahedron4<double>;
  using Coord = typename Lagrange::GlobalCoord;
};
TEST_F(TestLagrangeTetrahedron4, OnStandardElement) {
  auto tetra = Lagrange{
    Coord(0, 0, 0), Coord(3, 0, 0), Coord(0, 3, 0), Coord(0, 0, 3)
  };
  static_assert(tetra.CellDim() == 3);
  static_assert(tetra.PhysDim() == 3);
  EXPECT_EQ(tetra.center(), Coord(0.75, 0.75, 0.75));
  EXPECT_EQ(tetra.LocalToGlobal(1, 0, 0), Coord(0, 0, 0));
  EXPECT_EQ(tetra.LocalToGlobal(0, 1, 0), Coord(3, 0, 0));
  EXPECT_EQ(tetra.LocalToGlobal(0, 0, 1), Coord(0, 3, 0));
  EXPECT_EQ(tetra.LocalToGlobal(0, 0, 0), Coord(0, 0, 3));
  EXPECT_EQ(tetra.GlobalToLocal(0, 0, 0), Coord(1, 0, 0));
  EXPECT_EQ(tetra.GlobalToLocal(3, 0, 0), Coord(0, 1, 0));
  EXPECT_EQ(tetra.GlobalToLocal(0, 3, 0), Coord(0, 0, 1));
  EXPECT_EQ(tetra.GlobalToLocal(0, 0, 3), Coord(0, 0, 0));
}
TEST_F(TestLagrangeTetrahedron4, SortNodesOnFace) {
  using mini::lagrange::SortNodesOnFace;
  auto cell = Lagrange{
    Coord(0, 0, 0), Coord(3, 0, 0), Coord(0, 3, 0), Coord(0, 0, 3)
  };
  int face_n_node = 3;
  // test the version without conversion:
  {
    using Vector = std::vector<size_t>;
    Vector cell_nodes{ 11, 22, 33, 44 }, face_nodes, face_nodes_expect;
    face_nodes = { 11, 22, 33, 0 };
    face_nodes_expect = { 11, 33, 22, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 44, 22, 0 };
    face_nodes_expect = { 11, 22, 44, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 33, 44, 0 };
    face_nodes_expect = { 11, 44, 33, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 22, 33, 44, 0 };
    face_nodes_expect = { 22, 33, 44, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
  // test the version with conversion:
  {
    using Vector = std::vector<short>;
    Vector cell_nodes{ 11, 22, 33, 44 }, face_nodes, face_nodes_expect;
    face_nodes = { 11, 22, 33, 0 };
    face_nodes_expect = { 11, 33, 22, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 44, 22, 0 };
    face_nodes_expect = { 11, 22, 44, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 33, 44, 0 };
    face_nodes_expect = { 11, 44, 33, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 22, 33, 44, 0 };
    face_nodes_expect = { 22, 33, 44, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
