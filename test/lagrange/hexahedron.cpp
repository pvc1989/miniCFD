//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include <cmath>

#include "mini/lagrange/cell.hpp"
#include "mini/lagrange/hexahedron.hpp"

#include "gtest/gtest.h"

class TestLagrangeHexahedron8 : public ::testing::Test {
 protected:
  using Lagrange = mini::lagrange::Hexahedron8<double>;
  using Coord = typename Lagrange::GlobalCoord;
};
TEST_F(TestLagrangeHexahedron8, CoordinateMap) {
  auto hexa = Lagrange {
    Coord(-10, -10, -10), Coord(+10, -10, -10),
    Coord(+10, +10, -10), Coord(-10, +10, -10),
    Coord(-10, -10, +10), Coord(+10, -10, +10),
    Coord(+10, +10, +10), Coord(-10, +10, +10)
  };
  static_assert(hexa.CellDim() == 3);
  static_assert(hexa.PhysDim() == 3);
  EXPECT_EQ(hexa.LocalToGlobal(1, 1, 1), Coord(10, 10, 10));
  EXPECT_EQ(hexa.LocalToGlobal(1.5, 1.5, 1.5), Coord(15, 15, 15));
  EXPECT_EQ(hexa.LocalToGlobal(3, 4, 5), Coord(30, 40, 50));
  EXPECT_EQ(hexa.GlobalToLocal(30, 40, 20), Coord(3, 4, 2));
  EXPECT_EQ(hexa.GlobalToLocal(40, 55, 25), Coord(4, 5.5, 2.5));
  EXPECT_EQ(hexa.GlobalToLocal(70, 130, 60), Coord(7, 13, 6));
  mini::lagrange::Cell<typename Lagrange::Real> &cell = hexa;
  // test the partition-of-unity property:
  std::srand(31415926);
  auto rand = [](){ return -1 + 2.0 * std::rand() / (1.0 + RAND_MAX); };
  for (int i = 0; i < 1'000'000; ++i) {
    auto x = rand(), y = rand(), z = rand();
    auto shapes = cell.LocalToShapeFunctions(x, y, z);
    auto sum = std::accumulate(shapes.begin(), shapes.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-15);
  }
  // test the Kronecker-delta and property:
  for (int i = 0, n = cell.CountNodes(); i < n; ++i) {
    auto local_i = cell.GetLocalCoord(i);
    auto shapes = cell.LocalToShapeFunctions(local_i);
    for (int j = 0; j < n; ++j) {
      EXPECT_EQ(shapes[j], i == j);
    }
  }
}
TEST_F(TestLagrangeHexahedron8, SortNodesOnFace) {
  using mini::lagrange::SortNodesOnFace;
  auto cell = Lagrange{
    Coord(-10, -10, -10), Coord(+10, -10, -10),
    Coord(+10, +10, -10), Coord(-10, +10, -10),
    Coord(-10, -10, +10), Coord(+10, -10, +10),
    Coord(+10, +10, +10), Coord(-10, +10, +10)
  };
  int face_n_node = 4;
  // test the version without conversion:
  {
    using Vector = std::vector<size_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66, 77, 88, 00 };
    Vector face_nodes, face_nodes_expect;
    face_nodes = { 11, 22, 33, 44, 0 };
    face_nodes_expect = { 11, 44, 33, 22, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 22, 55, 66, 0 };
    face_nodes_expect = { 11, 22, 66, 55, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 44, 55, 88, 0 };
    face_nodes_expect = { 11, 55, 88, 44, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 22, 33, 66, 77, 0 };
    face_nodes_expect = { 22, 33, 77, 66, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 33, 44, 77, 88, 0 };
    face_nodes_expect = { 33, 44, 88, 77, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 66, 77, 88, 0 };
    face_nodes_expect = { 55, 66, 77, 88, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
  // test the version with conversion:
  {
    using Vector = std::vector<short>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66, 77, 88, 00 };
    Vector face_nodes, face_nodes_expect;
    face_nodes = { 11, 22, 33, 44, 0 };
    face_nodes_expect = { 11, 44, 33, 22, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 22, 55, 66, 0 };
    face_nodes_expect = { 11, 22, 66, 55, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 44, 55, 88, 0 };
    face_nodes_expect = { 11, 55, 88, 44, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 22, 33, 66, 77, 0 };
    face_nodes_expect = { 22, 33, 77, 66, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 33, 44, 77, 88, 0 };
    face_nodes_expect = { 33, 44, 88, 77, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 66, 77, 88, 0 };
    face_nodes_expect = { 55, 66, 77, 88, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
