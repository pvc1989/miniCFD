//  Copyright 2023 PEI Weicheng

#include <vector>
#include <algorithm>
#include <cstdlib>

#include "mini/lagrange/cell.hpp"
#include "mini/lagrange/pyramid.hpp"

#include "gtest/gtest.h"

class TestLagrangePyramid5 : public ::testing::Test {
 protected:
  static constexpr int kPoints = 24;
  using Lagrange = mini::lagrange::Pyramid5<double>;
  using Coord = typename Lagrange::Global;
};
TEST_F(TestLagrangePyramid5, CoordinateMap) {
  auto lagrange = Lagrange{
    Coord(-2, -2, 0), Coord(2, -2, 0), Coord(2, 2, 0), Coord(-2, 2, 0),
    Coord(0, 0, 4)
  };
  static_assert(lagrange.CellDim() == 3);
  static_assert(lagrange.PhysDim() == 3);
  EXPECT_EQ(lagrange.center(), Coord(0, 0, 2));
  EXPECT_EQ(lagrange.LocalToGlobal(1, 0, 0), Coord(1, 0, 2));
  EXPECT_EQ(lagrange.LocalToGlobal(0, 1, 0), Coord(0, 1, 2));
  EXPECT_EQ(lagrange.LocalToGlobal(0, 0, 1), Coord(0, 0, 4));
  EXPECT_EQ(lagrange.GlobalToLocal(lagrange.GetGlobalCoord(0)),
                                   lagrange.GetLocalCoord(0));
  EXPECT_EQ(lagrange.GlobalToLocal(lagrange.GetGlobalCoord(1)),
                                   lagrange.GetLocalCoord(1));
  EXPECT_EQ(lagrange.GlobalToLocal(lagrange.GetGlobalCoord(2)),
                                   lagrange.GetLocalCoord(2));
  EXPECT_EQ(lagrange.GlobalToLocal(lagrange.GetGlobalCoord(3)),
                                   lagrange.GetLocalCoord(3));
  EXPECT_NE(lagrange.GlobalToLocal(lagrange.GetGlobalCoord(4)),
                                   lagrange.GetLocalCoord(4));
  mini::lagrange::Cell<typename Lagrange::Real> &cell = lagrange;
  // test the partition-of-unity property:
  std::srand(31415926);
  auto rand = [](){ return std::rand() / (1.0 + RAND_MAX); };
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
TEST_F(TestLagrangePyramid5, SortNodesOnFace) {
  using mini::lagrange::SortNodesOnFace;
  auto cell = Lagrange{
    Coord(-2, -2, 0), Coord(2, -2, 0), Coord(2, 2, 0), Coord(-2, 2, 0),
    Coord(0, 0, 4)
  };
  // test the version without conversion:
  {
    using Vector = std::vector<size_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55 }, face_nodes, face_nodes_expect;
    int face_n_node = 3;
    face_nodes = { 55, 11, 22, 0 };
    face_nodes_expect = { 11, 22, 55, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 22, 33, 0 };
    face_nodes_expect = { 22, 33, 55, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 33, 44, 0 };
    face_nodes_expect = { 33, 44, 55, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 44, 11, 0 };
    face_nodes_expect = { 44, 11, 55, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_n_node = 4;
    face_nodes = { 22, 33, 44, 11, 0 };
    face_nodes_expect = { 11, 44, 33, 22, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
  // test the version with conversion:
  {
    using Vector = std::vector<short>;
    Vector cell_nodes{ 11, 22, 33, 44, 55 }, face_nodes, face_nodes_expect;
    int face_n_node = 3;
    face_nodes = { 55, 11, 22, 0 };
    face_nodes_expect = { 11, 22, 55, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 22, 33, 0 };
    face_nodes_expect = { 22, 33, 55, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 33, 44, 0 };
    face_nodes_expect = { 33, 44, 55, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 44, 11, 0 };
    face_nodes_expect = { 44, 11, 55, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_n_node = 4;
    face_nodes = { 22, 33, 44, 11, 0 };
    face_nodes_expect = { 11, 44, 33, 22, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
