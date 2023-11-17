//  Copyright 2023 PEI Weicheng

#include <vector>
#include <algorithm>
#include <cstdlib>

#include "mini/lagrange/cell.hpp"
#include "mini/lagrange/wedge.hpp"
#include "mini/lagrange/triangle.hpp"
#include "mini/lagrange/quadrangle.hpp"

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
  // test consistency with shape functions on Triangle3's:
  {
    std::vector<std::vector<int>> c_lists{
        { 0, 2, 1 }, { 3, 4, 5 }
    };
    for (auto c_list : c_lists) {
      auto c2f = std::vector<int>(cell.CountNodes(), -1);
      int f = 0;
      for (int c : c_list) { c2f[c] = f++; }
      auto face = mini::lagrange::Triangle3<double, 3>{
        cell.GetGlobalCoord(c_list[0]), cell.GetGlobalCoord(c_list[1]),
        cell.GetGlobalCoord(c_list[2]),
      };
      for (int i = 0; i < 1000; ++i) {
        auto x = rand(), y = rand();
        auto global = face.LocalToGlobal(x, y);
        auto local = cell.GlobalToLocal(global);
        auto cell_shapes = cell.LocalToShapeFunctions(local);
        auto face_shapes = face.LocalToShapeFunctions(x, y);
        for (int c = 0; c < cell.CountNodes(); c++) {
          int f = c2f[c];
          EXPECT_NEAR(cell_shapes[c], f < 0 ? 0 : face_shapes[f], 1e-16);
        }
      }
    }
  }
  // test consistency with shape functions on Quadrangle4's:
  {
    std::vector<std::vector<int>> c_lists{
        { 0, 1, 4, 3 }, { 1, 2, 5, 4 }, { 0, 3, 5, 2 },
    };
    for (auto c_list : c_lists) {
      auto c2f = std::vector<int>(cell.CountNodes(), -1);
      int f = 0;
      for (int c : c_list) { c2f[c] = f++; }
      auto face = mini::lagrange::Quadrangle4<double, 3>{
        cell.GetGlobalCoord(c_list[0]), cell.GetGlobalCoord(c_list[1]),
        cell.GetGlobalCoord(c_list[2]), cell.GetGlobalCoord(c_list[3]),
      };
      for (int i = 0; i < 1000; ++i) {
        auto x = rand() * 2 - 1, y = rand() * 2 - 1;
        auto global = face.LocalToGlobal(x, y);
        auto local = cell.GlobalToLocal(global);
        auto cell_shapes = cell.LocalToShapeFunctions(local);
        auto face_shapes = face.LocalToShapeFunctions(x, y);
        for (int c = 0; c < cell.CountNodes(); c++) {
          int f = c2f[c];
          EXPECT_NEAR(cell_shapes[c], f < 0 ? 0 : face_shapes[f], 1e-15);
        }
      }
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
    using Vector = std::vector<int16_t>;
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

class TestLagrangeWedge15 : public ::testing::Test {
 protected:
  using Lagrange = mini::lagrange::Wedge15<double>;
  using Coord = typename Lagrange::Global;
};
TEST_F(TestLagrangeWedge15, CoordinateMap) {
  auto lagrange = Lagrange{
    Coord(0, 0, -3), Coord(3, 0, -3), Coord(0, 3, -3),
    Coord(0, 0, +3), Coord(3, 0, +3), Coord(0, 3, +3),
    Coord(1.5, 0, -3), Coord(1.5, 1.5, -3), Coord(0, 1.5, -3),
    Coord(0, 0, 0), Coord(3, 0, 0), Coord(0, 3, 0),
    Coord(1.5, 0, +3), Coord(1.5, 1.5, +3), Coord(0, 1.5, +3),
  };
  static_assert(lagrange.CellDim() == 3);
  static_assert(lagrange.PhysDim() == 3);
  EXPECT_EQ(lagrange.CountCorners(), 6);
  EXPECT_EQ(lagrange.CountNodes(), 15);
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
    EXPECT_NEAR(sum, 1.0, 1e-14);
    auto grads = cell.LocalToShapeGradients(x, y, z);
    Coord grads_sum
        = std::accumulate(grads.begin(), grads.end(), Coord(0, 0, 0));
    EXPECT_NEAR(grads_sum.norm(), 0.0, 1e-14);
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
  // test consistency with shape functions on Triangle6's:
  {
    std::vector<std::vector<int>> c_lists{
        { 0, 2, 1, 8, 7, 6 }, { 3, 4, 5, 12, 13, 14 }
    };
    for (auto c_list : c_lists) {
      auto c2f = std::vector<int>(cell.CountNodes(), -1);
      int f = 0;
      for (int c : c_list) { c2f[c] = f++; }
      auto face = mini::lagrange::Triangle6<double, 3>{
        cell.GetGlobalCoord(c_list[0]), cell.GetGlobalCoord(c_list[1]),
        cell.GetGlobalCoord(c_list[2]), cell.GetGlobalCoord(c_list[3]),
        cell.GetGlobalCoord(c_list[4]), cell.GetGlobalCoord(c_list[5]),
      };
      for (int i = 0; i < 1000; ++i) {
        auto x = rand(), y = rand();
        auto global = face.LocalToGlobal(x, y);
        auto local = cell.GlobalToLocal(global);
        auto cell_shapes = cell.LocalToShapeFunctions(local);
        auto face_shapes = face.LocalToShapeFunctions(x, y);
        for (int c = 0; c < cell.CountNodes(); c++) {
          int f = c2f[c];
          EXPECT_NEAR(cell_shapes[c], f < 0 ? 0 : face_shapes[f], 1e-14);
        }
      }
    }
  }
  // test consistency with shape functions on Quadrangle8's:
  {
    std::vector<std::vector<int>> c_lists{
        { 0, 1, 4, 3, 6, 10, 12, 9 }, { 1, 2, 5, 4, 7, 11, 13, 10 },
        { 0, 3, 5, 2, 9, 14, 11, 8 },
    };
    for (auto c_list : c_lists) {
      auto c2f = std::vector<int>(cell.CountNodes(), -1);
      int f = 0;
      for (int c : c_list) { c2f[c] = f++; }
      auto face = mini::lagrange::Quadrangle8<double, 3>{
        cell.GetGlobalCoord(c_list[0]), cell.GetGlobalCoord(c_list[1]),
        cell.GetGlobalCoord(c_list[2]), cell.GetGlobalCoord(c_list[3]),
        cell.GetGlobalCoord(c_list[4]), cell.GetGlobalCoord(c_list[5]),
        cell.GetGlobalCoord(c_list[6]), cell.GetGlobalCoord(c_list[7]),
      };
      for (int i = 0; i < 1000; ++i) {
        auto x = rand() * 2 - 1, y = rand() * 2 - 1;
        auto global = face.LocalToGlobal(x, y);
        auto local = cell.GlobalToLocal(global);
        auto cell_shapes = cell.LocalToShapeFunctions(local);
        auto face_shapes = face.LocalToShapeFunctions(x, y);
        for (int c = 0; c < cell.CountNodes(); c++) {
          int f = c2f[c];
          EXPECT_NEAR(cell_shapes[c], f < 0 ? 0 : face_shapes[f], 1e-14);
        }
      }
    }
  }
}
TEST_F(TestLagrangeWedge15, SortNodesOnFace) {
  using mini::lagrange::SortNodesOnFace;
  auto cell = Lagrange{
    Coord(0, 0, -3), Coord(3, 0, -3), Coord(0, 3, -3),
    Coord(0, 0, +3), Coord(3, 0, +3), Coord(0, 3, +3),
    Coord(1.5, 0, -3), Coord(1.5, 1.5, -3), Coord(0, 1.5, -3),
    Coord(0, 0, 0), Coord(3, 0, 0), Coord(0, 3, 0),
    Coord(1.5, 0, +3), Coord(1.5, 1.5, +3), Coord(0, 1.5, +3),
  };
  // test the version without conversion:
  {
    using Vector = std::vector<size_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66, 77, 88, 99, 1010,
      1111, 1212, 1313, 1414, 1515, 00 };
    Vector face_nodes, face_nodes_expect;
    int face_n_node = 6;
    face_nodes = { 55, 44, 66, 1313, 1414, 1515, 0 };
    face_nodes_expect = { 44, 55, 66, 1313, 1414, 1515, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 22, 33, 77, 88, 99, 0 };
    face_nodes_expect = { 11, 33, 22, 99, 88, 77, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_n_node = 8;
    face_nodes = { 22, 11, 44, 55, 77, 1010, 1111, 1313, 0 };
    face_nodes_expect = { 11, 22, 55, 44, 77, 1111, 1313, 1010, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 33, 22, 66, 1414, 1212, 1111, 88, 0 };
    face_nodes_expect = { 22, 33, 66, 55, 88, 1212, 1414, 1111, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 33, 66, 44, 11, 1515, 1212, 1010, 99, 0 };
    face_nodes_expect = { 11, 44, 66, 33, 1010, 1515, 1212, 99, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
  // test the version with conversion:
  {
    using Vector = std::vector<int16_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66, 77, 88, 99, 1010,
      1111, 1212, 1313, 1414, 1515, 00 };
    Vector face_nodes, face_nodes_expect;
    int face_n_node = 6;
    face_nodes = { 55, 44, 66, 1313, 1414, 1515, 0 };
    face_nodes_expect = { 44, 55, 66, 1313, 1414, 1515, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 22, 33, 77, 88, 99, 0 };
    face_nodes_expect = { 11, 33, 22, 99, 88, 77, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_n_node = 8;
    face_nodes = { 22, 11, 44, 55, 77, 1010, 1111, 1313, 0 };
    face_nodes_expect = { 11, 22, 55, 44, 77, 1111, 1313, 1010, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 33, 22, 66, 1414, 1212, 1111, 88, 0 };
    face_nodes_expect = { 22, 33, 66, 55, 88, 1212, 1414, 1111, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 33, 66, 44, 11, 1515, 1212, 1010, 99, 0 };
    face_nodes_expect = { 11, 44, 66, 33, 1010, 1515, 1212, 99, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
}

class TestLagrangeWedge18 : public ::testing::Test {
 protected:
  using Lagrange = mini::lagrange::Wedge18<double>;
  using Coord = typename Lagrange::Global;
};
TEST_F(TestLagrangeWedge18, CoordinateMap) {
  auto lagrange = Lagrange{
    Coord(0, 0, -3), Coord(3, 0, -3), Coord(0, 3, -3),
    Coord(0, 0, +3), Coord(3, 0, +3), Coord(0, 3, +3),
    Coord(1.5, 0, -3), Coord(1.5, 1.5, -3), Coord(0, 1.5, -3),
    Coord(0, 0, 0), Coord(3, 0, 0), Coord(0, 3, 0),
    Coord(1.5, 0, +3), Coord(1.5, 1.5, +3), Coord(0, 1.5, +3),
    Coord(1.5, 0, 0), Coord(1.5, 1.5, 0), Coord(0, 1.5, 0),
  };
  static_assert(lagrange.CellDim() == 3);
  static_assert(lagrange.PhysDim() == 3);
  EXPECT_EQ(lagrange.CountCorners(), 6);
  EXPECT_EQ(lagrange.CountNodes(), 18);
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
    EXPECT_NEAR(sum, 1.0, 1e-14);
    auto grads = cell.LocalToShapeGradients(x, y, z);
    Coord grads_sum
        = std::accumulate(grads.begin(), grads.end(), Coord(0, 0, 0));
    EXPECT_NEAR(grads_sum.norm(), 0.0, 1e-14);
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
  // test consistency with shape functions on Triangle6's:
  {
    std::vector<std::vector<int>> c_lists{
        { 0, 2, 1, 8, 7, 6 }, { 3, 4, 5, 12, 13, 14 }
    };
    for (auto c_list : c_lists) {
      auto c2f = std::vector<int>(cell.CountNodes(), -1);
      int f = 0;
      for (int c : c_list) { c2f[c] = f++; }
      auto face = mini::lagrange::Triangle6<double, 3>{
        cell.GetGlobalCoord(c_list[0]), cell.GetGlobalCoord(c_list[1]),
        cell.GetGlobalCoord(c_list[2]), cell.GetGlobalCoord(c_list[3]),
        cell.GetGlobalCoord(c_list[4]), cell.GetGlobalCoord(c_list[5]),
      };
      for (int i = 0; i < 1000; ++i) {
        auto x = rand(), y = rand();
        auto global = face.LocalToGlobal(x, y);
        auto local = cell.GlobalToLocal(global);
        auto cell_shapes = cell.LocalToShapeFunctions(local);
        auto face_shapes = face.LocalToShapeFunctions(x, y);
        for (int c = 0; c < cell.CountNodes(); c++) {
          int f = c2f[c];
          EXPECT_NEAR(cell_shapes[c], f < 0 ? 0 : face_shapes[f], 1e-13);
        }
      }
    }
  }
  // test consistency with shape functions on Quadrangle9's:
  {
    std::vector<std::vector<int>> c_lists{
        { 0, 1, 4, 3, 6, 10, 12, 9, 15 }, { 1, 2, 5, 4, 7, 11, 13, 10, 16 },
        { 0, 3, 5, 2, 9, 14, 11, 8, 17 },
    };
    for (auto c_list : c_lists) {
      auto c2f = std::vector<int>(cell.CountNodes(), -1);
      int f = 0;
      for (int c : c_list) { c2f[c] = f++; }
      auto face = mini::lagrange::Quadrangle9<double, 3>{
        cell.GetGlobalCoord(c_list[0]), cell.GetGlobalCoord(c_list[1]),
        cell.GetGlobalCoord(c_list[2]), cell.GetGlobalCoord(c_list[3]),
        cell.GetGlobalCoord(c_list[4]), cell.GetGlobalCoord(c_list[5]),
        cell.GetGlobalCoord(c_list[6]), cell.GetGlobalCoord(c_list[7]),
        cell.GetGlobalCoord(c_list[8]),
      };
      for (int i = 0; i < 1000; ++i) {
        auto x = rand() * 2 - 1, y = rand() * 2 - 1;
        auto global = face.LocalToGlobal(x, y);
        auto local = cell.GlobalToLocal(global);
        auto cell_shapes = cell.LocalToShapeFunctions(local);
        auto face_shapes = face.LocalToShapeFunctions(x, y);
        for (int c = 0; c < cell.CountNodes(); c++) {
          int f = c2f[c];
          EXPECT_NEAR(cell_shapes[c], f < 0 ? 0 : face_shapes[f], 1e-14);
        }
      }
    }
  }
}
TEST_F(TestLagrangeWedge18, SortNodesOnFace) {
  using mini::lagrange::SortNodesOnFace;
  auto cell = Lagrange{
    Coord(0, 0, -3), Coord(3, 0, -3), Coord(0, 3, -3),
    Coord(0, 0, +3), Coord(3, 0, +3), Coord(0, 3, +3),
    Coord(1.5, 0, -3), Coord(1.5, 1.5, -3), Coord(0, 1.5, -3),
    Coord(0, 0, 0), Coord(3, 0, 0), Coord(0, 3, 0),
    Coord(1.5, 0, +3), Coord(1.5, 1.5, +3), Coord(0, 1.5, +3),
    Coord(1.5, 0, 0), Coord(1.5, 1.5, 0), Coord(0, 1.5, 0),
  };
  // test the version without conversion:
  {
    using Vector = std::vector<size_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66, 77, 88, 99, 1010,
      1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 00 };
    Vector face_nodes, face_nodes_expect;
    int face_n_node = 6;
    face_nodes = { 55, 44, 66, 1313, 1414, 1515, 0 };
    face_nodes_expect = { 44, 55, 66, 1313, 1414, 1515, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 22, 33, 77, 88, 99, 0 };
    face_nodes_expect = { 11, 33, 22, 99, 88, 77, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_n_node = 9;
    face_nodes = { 22, 11, 44, 55, 1616, 77, 1010, 1111, 1313, 0 };
    face_nodes_expect = { 11, 22, 55, 44, 77, 1111, 1313, 1010, 1616, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 33, 22, 66, 1717, 1414, 1212, 1111, 88, 0 };
    face_nodes_expect = { 22, 33, 66, 55, 88, 1212, 1414, 1111, 1717, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 33, 66, 44, 11, 1818, 1515, 1212, 1010, 99, 0 };
    face_nodes_expect = { 11, 44, 66, 33, 1010, 1515, 1212, 99, 1818, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
  // test the version with conversion:
  {
    using Vector = std::vector<int16_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66, 77, 88, 99, 1010,
      1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 00 };
    Vector face_nodes, face_nodes_expect;
    int face_n_node = 6;
    face_nodes = { 55, 44, 66, 1313, 1414, 1515, 0 };
    face_nodes_expect = { 44, 55, 66, 1313, 1414, 1515, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 22, 33, 77, 88, 99, 0 };
    face_nodes_expect = { 11, 33, 22, 99, 88, 77, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_n_node = 9;
    face_nodes = { 22, 11, 44, 55, 1616, 77, 1010, 1111, 1313, 0 };
    face_nodes_expect = { 11, 22, 55, 44, 77, 1111, 1313, 1010, 1616, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 33, 22, 66, 1717, 1414, 1212, 1111, 88, 0 };
    face_nodes_expect = { 22, 33, 66, 55, 88, 1212, 1414, 1111, 1717, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 33, 66, 44, 11, 1818, 1515, 1212, 1010, 99, 0 };
    face_nodes_expect = { 11, 44, 66, 33, 1010, 1515, 1212, 99, 1818, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
