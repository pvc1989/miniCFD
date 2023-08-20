//  Copyright 2023 PEI Weicheng

#include <vector>
#include <algorithm>
#include <cstdlib>

#include "mini/lagrange/cell.hpp"
#include "mini/lagrange/wedge.hpp"

#include "gtest/gtest.h"

class TestLagrangeWedge6 : public ::testing::Test {
 protected:
  using Lagrange = mini::lagrange::Wedge6<double>;
  using Coord = typename Lagrange::Global;
};
TEST_F(TestLagrangeWedge6, CoordinateMap) {
  auto lagrange = Lagrange{
    Coord(0, 0, -3), Coord(3, 0, -3), Coord(0, 3, -3),
    Coord(0, 0, +3), Coord(3, 0, +3), Coord(0, 3, +3)
  };
  static_assert(lagrange.CellDim() == 3);
  static_assert(lagrange.PhysDim() == 3);
  EXPECT_EQ(lagrange.CountCorners(), 6);
  EXPECT_EQ(lagrange.CountNodes(), 6);
  EXPECT_NEAR((lagrange.center() - Coord(1, 1, 0)).norm(), 0, 1e-15);
  EXPECT_EQ(lagrange.LocalToGlobal(1, 0, 0), Coord(0, 0, 0));
  EXPECT_EQ(lagrange.LocalToGlobal(0, 1, 0), Coord(3, 0, 0));
  EXPECT_EQ(lagrange.LocalToGlobal(0, 0, 0), Coord(0, 3, 0));
  EXPECT_EQ(lagrange.GlobalToLocal(lagrange.GetGlobalCoord(0)),
                                   lagrange.GetLocalCoord(0));
  EXPECT_EQ(lagrange.GlobalToLocal(lagrange.GetGlobalCoord(1)),
                                   lagrange.GetLocalCoord(1));
  EXPECT_EQ(lagrange.GlobalToLocal(lagrange.GetGlobalCoord(2)),
                                   lagrange.GetLocalCoord(2));
  EXPECT_EQ(lagrange.GlobalToLocal(lagrange.GetGlobalCoord(3)),
                                   lagrange.GetLocalCoord(3));
  EXPECT_EQ(lagrange.GlobalToLocal(lagrange.GetGlobalCoord(4)),
                                   lagrange.GetLocalCoord(4));
  EXPECT_EQ(lagrange.GlobalToLocal(lagrange.GetGlobalCoord(5)),
                                   lagrange.GetLocalCoord(5));
  mini::lagrange::Cell<typename Lagrange::Real> &cell = lagrange;
  // test the partition-of-unity property:
  std::srand(31415926);
  auto rand = [](){ return std::rand() / (1.0 + RAND_MAX); };
  for (int i = 0; i < 1000; ++i) {
    auto x = rand(), y = rand(), z = 2 * rand() - 1;
    auto shapes = cell.LocalToShapeFunctions(x, y, z);
    auto sum = std::accumulate(shapes.begin(), shapes.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-15);
    auto grads = cell.LocalToShapeGradients(x, y, z);
    Coord grads_sum
        = std::accumulate(grads.begin(), grads.end(), Coord(0, 0, 0));
    EXPECT_NEAR(grads_sum.norm(), 0.0, 1e-15);
    // compare gradients with O(h^2) finite difference derivatives
    int X{0}, Y{1}, Z{2};
    auto h = 1e-6;
    int n_node = cell.CountNodes();
    auto left = cell.LocalToShapeFunctions(x - h, y, z);
    auto right = cell.LocalToShapeFunctions(x + h, y, z);
    for (int i_node = 0; i_node < n_node; ++i_node) {
      grads[i_node][X] -= (right[i_node] - left[i_node]) / (2 * h);
    }
    left = cell.LocalToShapeFunctions(x, y - h, z);
    right = cell.LocalToShapeFunctions(x, y + h, z);
    for (int i_node = 0; i_node < n_node; ++i_node) {
      grads[i_node][Y] -= (right[i_node] - left[i_node]) / (2 * h);
    }
    left = cell.LocalToShapeFunctions(x, y, z - h);
    right = cell.LocalToShapeFunctions(x, y, z + h);
    for (int i_node = 0; i_node < n_node; ++i_node) {
      grads[i_node][Z] -= (right[i_node] - left[i_node]) / (2 * h);
      EXPECT_NEAR(grads[i_node].norm(), 0.0, 1e-9);
    }
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
TEST_F(TestLagrangeWedge6, SortNodesOnFace) {
  using mini::lagrange::SortNodesOnFace;
  auto cell = Lagrange{
    Coord(0, 0, -3), Coord(3, 0, -3), Coord(0, 3, -3),
    Coord(0, 0, +3), Coord(3, 0, +3), Coord(0, 3, +3)
  };
  // test the version without conversion:
  {
    using Vector = std::vector<size_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66 }, face_nodes, face_nodes_expect;
    int face_n_node = 3;
    face_nodes = { 55, 44, 66, 0 };
    face_nodes_expect = { 44, 55, 66, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 22, 33, 11, 0 };
    face_nodes_expect = { 11, 33, 22, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_n_node = 4;
    face_nodes = { 22, 11, 44, 55, 0 };
    face_nodes_expect = { 11, 22, 55, 44, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 33, 22, 66, 0 };
    face_nodes_expect = { 22, 33, 66, 55, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 33, 66, 44, 11, 0 };
    face_nodes_expect = { 11, 44, 66, 33, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
  // test the version with conversion:
  {
    using Vector = std::vector<short>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66 }, face_nodes, face_nodes_expect;
    int face_n_node = 3;
    face_nodes = { 55, 44, 66, 0 };
    face_nodes_expect = { 44, 55, 66, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 22, 33, 11, 0 };
    face_nodes_expect = { 11, 33, 22, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_n_node = 4;
    face_nodes = { 22, 11, 44, 55, 0 };
    face_nodes_expect = { 11, 22, 55, 44, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 33, 22, 66, 0 };
    face_nodes_expect = { 22, 33, 66, 55, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 33, 66, 44, 11, 0 };
    face_nodes_expect = { 11, 44, 66, 33, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
