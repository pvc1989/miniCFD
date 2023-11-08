//  Copyright 2023 PEI Weicheng

#include <numeric>

#include <cmath>

#include "mini/geometry/cell.hpp"
#include "mini/geometry/hexahedron.hpp"

#include "gtest/gtest.h"

class TestLagrangeHexahedron8 : public ::testing::Test {
 protected:
  using Lagrange = mini::geometry::Hexahedron8<double>;
  using Coord = typename Lagrange::Global;
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
  EXPECT_EQ(hexa.CountCorners(), 8);
  EXPECT_EQ(hexa.CountNodes(), 8);
  EXPECT_EQ(hexa.LocalToGlobal(1, 1, 1), Coord(10, 10, 10));
  EXPECT_EQ(hexa.LocalToGlobal(1.5, 1.5, 1.5), Coord(15, 15, 15));
  EXPECT_EQ(hexa.LocalToGlobal(3, 4, 5), Coord(30, 40, 50));
  EXPECT_EQ(hexa.GlobalToLocal(30, 40, 20), Coord(3, 4, 2));
  EXPECT_EQ(hexa.GlobalToLocal(40, 55, 25), Coord(4, 5.5, 2.5));
  EXPECT_EQ(hexa.GlobalToLocal(70, 130, 60), Coord(7, 13, 6));
  mini::geometry::Cell<typename Lagrange::Real> &cell = hexa;
  // test the partition-of-unity property:
  std::srand(31415926);
  auto rand = [](){ return -1 + 2.0 * std::rand() / (1.0 + RAND_MAX); };
  for (int i = 0; i < 1000; ++i) {
    auto x = rand(), y = rand(), z = rand();
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
TEST_F(TestLagrangeHexahedron8, SortNodesOnFace) {
  using mini::geometry::SortNodesOnFace;
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
    using Vector = std::vector<int16_t>;
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

class TestLagrangeHexahedron20 : public ::testing::Test {
 protected:
  using Lagrange = mini::geometry::Hexahedron20<double>;
  using Coord = typename Lagrange::Global;
};
TEST_F(TestLagrangeHexahedron20, CoordinateMap) {
  auto hexa = Lagrange {
    // corner nodes on the bottom face
    Coord(-10, -10, -10), Coord(+10, -10, -10),
    Coord(+10, +10, -10), Coord(-10, +10, -10),
    // corner nodes on the top face
    Coord(-10, -10, +10), Coord(+10, -10, +10),
    Coord(+10, +10, +10), Coord(-10, +10, +10),
    // mid-edge nodes on the bottom face
    Coord(0, -10, -10), Coord(+10, 0, -10),
    Coord(0, +10, -10), Coord(-10, 0, -10),
    // mid-edge nodes on vertical edges
    Coord(-10, -10, 0), Coord(+10, -10, 0),
    Coord(+10, +10, 0), Coord(-10, +10, 0),
    // mid-edge nodes on the top face
    Coord(0, -10, +10), Coord(+10, 0, +10),
    Coord(0, +10, +10), Coord(-10, 0, +10),
  };
  static_assert(hexa.CellDim() == 3);
  static_assert(hexa.PhysDim() == 3);
  EXPECT_EQ(hexa.CountCorners(), 8);
  EXPECT_EQ(hexa.CountNodes(), 20);
  EXPECT_EQ(hexa.LocalToGlobal(1, 1, 1), Coord(10, 10, 10));
  EXPECT_EQ(hexa.LocalToGlobal(1.5, 1.5, 1.5), Coord(15, 15, 15));
  EXPECT_EQ(hexa.LocalToGlobal(3, 4, 5), Coord(30, 40, 50));
  EXPECT_EQ(hexa.GlobalToLocal(30, 40, 20), Coord(3, 4, 2));
  EXPECT_EQ(hexa.GlobalToLocal(40, 55, 25), Coord(4, 5.5, 2.5));
  EXPECT_EQ(hexa.GlobalToLocal(70, 130, 60), Coord(7, 13, 6));
  mini::geometry::Cell<typename Lagrange::Real> &cell = hexa;
  // test the partition-of-unity property:
  std::srand(31415926);
  auto rand = [](){ return -1 + 2.0 * std::rand() / (1.0 + RAND_MAX); };
  for (int i = 0; i < 1000; ++i) {
    auto x = rand(), y = rand(), z = rand();
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
TEST_F(TestLagrangeHexahedron20, SortNodesOnFace) {
  using mini::geometry::SortNodesOnFace;
  auto cell = Lagrange{
    // corner nodes on the bottom face
    Coord(-10, -10, -10), Coord(+10, -10, -10),
    Coord(+10, +10, -10), Coord(-10, +10, -10),
    // corner nodes on the top face
    Coord(-10, -10, +10), Coord(+10, -10, +10),
    Coord(+10, +10, +10), Coord(-10, +10, +10),
    // mid-edge nodes on the bottom face
    Coord(0, -10, -10), Coord(+10, 0, -10),
    Coord(0, +10, -10), Coord(-10, 0, -10),
    // mid-edge nodes on vertical edges
    Coord(-10, -10, 0), Coord(+10, -10, 0),
    Coord(+10, +10, 0), Coord(-10, +10, 0),
    // mid-edge nodes on the top face
    Coord(0, -10, +10), Coord(+10, 0, +10),
    Coord(0, +10, +10), Coord(-10, 0, +10),
  };
  int face_n_node = 8;
  // test the version without conversion:
  {
    using Vector = std::vector<size_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66, 77, 88, 99, 1010,
      1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020, 00 };
    Vector face_nodes, face_nodes_expect;
    face_nodes = { 11, 22, 33, 44, 1111, 1010, 1212, 99, 0 };
    face_nodes_expect = { 11, 44, 33, 22, 1212, 1111, 1010, 99, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 22, 55, 66, 1313, 1414, 1717, 99, 0 };
    face_nodes_expect = { 11, 22, 66, 55, 99, 1414, 1717, 1313, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 44, 55, 88, 1212, 1313, 2020, 1616, 0 };
    face_nodes_expect = { 11, 55, 88, 44, 1313, 2020, 1616, 1212, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 22, 33, 66, 77, 1818, 1515, 1414, 1010, 0 };
    face_nodes_expect = { 22, 33, 77, 66, 1010, 1515, 1818, 1414, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 33, 44, 77, 88, 1919, 1616, 1515, 1111, 0 };
    face_nodes_expect = { 33, 44, 88, 77, 1111, 1616, 1919, 1515, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 66, 77, 88, 2020, 1919, 1818, 1717, 0 };
    face_nodes_expect = { 55, 66, 77, 88, 1717, 1818, 1919, 2020, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
  // test the version with conversion:
  {
    using Vector = std::vector<int16_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66, 77, 88, 99, 1010,
      1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020, 00 };
    Vector face_nodes, face_nodes_expect;
    face_nodes = { 11, 22, 33, 44, 1111, 1010, 1212, 99, 0 };
    face_nodes_expect = { 11, 44, 33, 22, 1212, 1111, 1010, 99, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 22, 55, 66, 1313, 1414, 1717, 99, 0 };
    face_nodes_expect = { 11, 22, 66, 55, 99, 1414, 1717, 1313, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 44, 55, 88, 1212, 1313, 2020, 1616, 0 };
    face_nodes_expect = { 11, 55, 88, 44, 1313, 2020, 1616, 1212, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 22, 33, 66, 77, 1818, 1515, 1414, 1010, 0 };
    face_nodes_expect = { 22, 33, 77, 66, 1010, 1515, 1818, 1414, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 33, 44, 77, 88, 1919, 1616, 1515, 1111, 0 };
    face_nodes_expect = { 33, 44, 88, 77, 1111, 1616, 1919, 1515, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 66, 77, 88, 2020, 1919, 1818, 1717, 0 };
    face_nodes_expect = { 55, 66, 77, 88, 1717, 1818, 1919, 2020, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
}

class TestLagrangeHexahedron27 : public ::testing::Test {
 protected:
  using Lagrange = mini::geometry::Hexahedron27<double>;
  using Coord = typename Lagrange::Global;
};
TEST_F(TestLagrangeHexahedron27, CoordinateMap) {
  auto hexa = Lagrange {
    // corner nodes on the bottom face
    Coord(-10, -10, -10), Coord(+10, -10, -10),
    Coord(+10, +10, -10), Coord(-10, +10, -10),
    // corner nodes on the top face
    Coord(-10, -10, +10), Coord(+10, -10, +10),
    Coord(+10, +10, +10), Coord(-10, +10, +10),
    // mid-edge nodes on the bottom face
    Coord(0, -10, -10), Coord(+10, 0, -10),
    Coord(0, +10, -10), Coord(-10, 0, -10),
    // mid-edge nodes on vertical edges
    Coord(-10, -10, 0), Coord(+10, -10, 0),
    Coord(+10, +10, 0), Coord(-10, +10, 0),
    // mid-edge nodes on the top face
    Coord(0, -10, +10), Coord(+10, 0, +10),
    Coord(0, +10, +10), Coord(-10, 0, +10),
    // mid-face nodes
    Coord(0, 0, -10),
    Coord(0, -10, 0), Coord(+10, 0, 0),
    Coord(0, +10, 0), Coord(-10, 0, 0),
    Coord(0, 0, +10),
    // center node
    Coord(0, 0, 0),
  };
  static_assert(hexa.CellDim() == 3);
  static_assert(hexa.PhysDim() == 3);
  EXPECT_EQ(hexa.CountCorners(), 8);
  EXPECT_EQ(hexa.CountNodes(), 27);
  EXPECT_EQ(hexa.LocalToGlobal(1, 1, 1), Coord(10, 10, 10));
  EXPECT_EQ(hexa.LocalToGlobal(1.5, 1.5, 1.5), Coord(15, 15, 15));
  EXPECT_EQ(hexa.LocalToGlobal(3, 4, 5), Coord(30, 40, 50));
  EXPECT_EQ(hexa.GlobalToLocal(30, 40, 20), Coord(3, 4, 2));
  EXPECT_EQ(hexa.GlobalToLocal(40, 55, 25), Coord(4, 5.5, 2.5));
  EXPECT_EQ(hexa.GlobalToLocal(70, 130, 60), Coord(7, 13, 6));
  mini::geometry::Cell<typename Lagrange::Real> &cell = hexa;
  // test the partition-of-unity property:
  std::srand(31415926);
  auto rand = [](){ return -1 + 2.0 * std::rand() / (1.0 + RAND_MAX); };
  for (int i = 0; i < 1000; ++i) {
    auto x = rand(), y = rand(), z = rand();
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
TEST_F(TestLagrangeHexahedron27, SortNodesOnFace) {
  using mini::geometry::SortNodesOnFace;
  auto cell = Lagrange{
    // corner nodes on the bottom face
    Coord(-10, -10, -10), Coord(+10, -10, -10),
    Coord(+10, +10, -10), Coord(-10, +10, -10),
    // corner nodes on the top face
    Coord(-10, -10, +10), Coord(+10, -10, +10),
    Coord(+10, +10, +10), Coord(-10, +10, +10),
    // mid-edge nodes on the bottom face
    Coord(0, -10, -10), Coord(+10, 0, -10),
    Coord(0, +10, -10), Coord(-10, 0, -10),
    // mid-edge nodes on vertical edges
    Coord(-10, -10, 0), Coord(+10, -10, 0),
    Coord(+10, +10, 0), Coord(-10, +10, 0),
    // mid-edge nodes on the top face
    Coord(0, -10, +10), Coord(+10, 0, +10),
    Coord(0, +10, +10), Coord(-10, 0, +10),
    // mid-face nodes
    Coord(0, 0, -10),
    Coord(0, -10, 0), Coord(+10, 0, 0),
    Coord(0, +10, 0), Coord(-10, 0, 0),
    Coord(0, 0, +10),
    // center node
    Coord(0, 0, 0),
  };
  int face_n_node = 9;
  // test the version without conversion:
  {
    using Vector = std::vector<size_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66, 77, 88, 99, 1010,
      1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020,
      2121, 2222, 2323, 2424, 2525, 2626, 2727, 00 };
    Vector face_nodes, face_nodes_expect;
    face_nodes = { 11, 22, 33, 44, 2121, 1111, 1010, 1212, 99, 0 };
    face_nodes_expect = { 11, 44, 33, 22, 1212, 1111, 1010, 99, 2121, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 22, 55, 66, 2222, 1313, 1414, 1717, 99, 0 };
    face_nodes_expect = { 11, 22, 66, 55, 99, 1414, 1717, 1313, 2222, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 44, 55, 88, 2525, 1212, 1313, 2020, 1616, 0 };
    face_nodes_expect = { 11, 55, 88, 44, 1313, 2020, 1616, 1212, 2525, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 22, 33, 66, 77, 2323, 1818, 1515, 1414, 1010, 0 };
    face_nodes_expect = { 22, 33, 77, 66, 1010, 1515, 1818, 1414, 2323, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 33, 44, 77, 88, 2424, 1919, 1616, 1515, 1111, 0 };
    face_nodes_expect = { 33, 44, 88, 77, 1111, 1616, 1919, 1515, 2424, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 66, 77, 88, 2626, 2020, 1919, 1818, 1717, 0 };
    face_nodes_expect = { 55, 66, 77, 88, 1717, 1818, 1919, 2020, 2626, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
  // test the version with conversion:
  {
    using Vector = std::vector<int16_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66, 77, 88, 99, 1010,
      1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020,
      2121, 2222, 2323, 2424, 2525, 2626, 2727, 00 };
    Vector face_nodes, face_nodes_expect;
    face_nodes = { 11, 22, 33, 44, 2121, 1111, 1010, 1212, 99, 0 };
    face_nodes_expect = { 11, 44, 33, 22, 1212, 1111, 1010, 99, 2121, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 22, 55, 66, 2222, 1313, 1414, 1717, 99, 0 };
    face_nodes_expect = { 11, 22, 66, 55, 99, 1414, 1717, 1313, 2222, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 44, 55, 88, 2525, 1212, 1313, 2020, 1616, 0 };
    face_nodes_expect = { 11, 55, 88, 44, 1313, 2020, 1616, 1212, 2525, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 22, 33, 66, 77, 2323, 1818, 1515, 1414, 1010, 0 };
    face_nodes_expect = { 22, 33, 77, 66, 1010, 1515, 1818, 1414, 2323, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 33, 44, 77, 88, 2424, 1919, 1616, 1515, 1111, 0 };
    face_nodes_expect = { 33, 44, 88, 77, 1111, 1616, 1919, 1515, 2424, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 66, 77, 88, 2626, 2020, 1919, 1818, 1717, 0 };
    face_nodes_expect = { 55, 66, 77, 88, 1717, 1818, 1919, 2020, 2626, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
}


class TestLagrangeHexahedron26 : public ::testing::Test {
 protected:
  using Lagrange = mini::geometry::Hexahedron26<double>;
  using Coord = typename Lagrange::Global;
};
TEST_F(TestLagrangeHexahedron26, CoordinateMap) {
  auto hexa = Lagrange {
    // corner nodes on the bottom face
    Coord(-10, -10, -10), Coord(+10, -10, -10),
    Coord(+10, +10, -10), Coord(-10, +10, -10),
    // corner nodes on the top face
    Coord(-10, -10, +10), Coord(+10, -10, +10),
    Coord(+10, +10, +10), Coord(-10, +10, +10),
    // mid-edge nodes on the bottom face
    Coord(0, -10, -10), Coord(+10, 0, -10),
    Coord(0, +10, -10), Coord(-10, 0, -10),
    // mid-edge nodes on vertical edges
    Coord(-10, -10, 0), Coord(+10, -10, 0),
    Coord(+10, +10, 0), Coord(-10, +10, 0),
    // mid-edge nodes on the top face
    Coord(0, -10, +10), Coord(+10, 0, +10),
    Coord(0, +10, +10), Coord(-10, 0, +10),
    // mid-face nodes
    Coord(0, 0, -10),
    Coord(0, -10, 0), Coord(+10, 0, 0),
    Coord(0, +10, 0), Coord(-10, 0, 0),
    Coord(0, 0, +10),
  };
  static_assert(hexa.CellDim() == 3);
  static_assert(hexa.PhysDim() == 3);
  EXPECT_EQ(hexa.CountCorners(), 8);
  EXPECT_EQ(hexa.CountNodes(), 26);
  EXPECT_EQ(hexa.LocalToGlobal(1, 1, 1), Coord(10, 10, 10));
  EXPECT_EQ(hexa.LocalToGlobal(1.5, 1.5, 1.5), Coord(15, 15, 15));
  EXPECT_EQ(hexa.LocalToGlobal(3, 4, 5), Coord(30, 40, 50));
  EXPECT_EQ(hexa.GlobalToLocal(30, 40, 20), Coord(3, 4, 2));
  EXPECT_EQ(hexa.GlobalToLocal(40, 55, 25), Coord(4, 5.5, 2.5));
  EXPECT_EQ(hexa.GlobalToLocal(70, 130, 60), Coord(7, 13, 6));
  mini::geometry::Cell<typename Lagrange::Real> &cell = hexa;
  // test the partition-of-unity property:
  std::srand(31415926);
  auto rand = [](){ return -1 + 2.0 * std::rand() / (1.0 + RAND_MAX); };
  for (int i = 0; i < 1000; ++i) {
    auto x = rand(), y = rand(), z = rand();
    auto shapes = cell.LocalToShapeFunctions(x, y, z);
    auto sum = std::accumulate(shapes.begin(), shapes.end(), 0.0);
    EXPECT_NEAR(sum, 1.0, 1e-15);
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
}
TEST_F(TestLagrangeHexahedron26, SortNodesOnFace) {
  using mini::geometry::SortNodesOnFace;
  auto cell = Lagrange{
    // corner nodes on the bottom face
    Coord(-10, -10, -10), Coord(+10, -10, -10),
    Coord(+10, +10, -10), Coord(-10, +10, -10),
    // corner nodes on the top face
    Coord(-10, -10, +10), Coord(+10, -10, +10),
    Coord(+10, +10, +10), Coord(-10, +10, +10),
    // mid-edge nodes on the bottom face
    Coord(0, -10, -10), Coord(+10, 0, -10),
    Coord(0, +10, -10), Coord(-10, 0, -10),
    // mid-edge nodes on vertical edges
    Coord(-10, -10, 0), Coord(+10, -10, 0),
    Coord(+10, +10, 0), Coord(-10, +10, 0),
    // mid-edge nodes on the top face
    Coord(0, -10, +10), Coord(+10, 0, +10),
    Coord(0, +10, +10), Coord(-10, 0, +10),
    // mid-face nodes
    Coord(0, 0, -10),
    Coord(0, -10, 0), Coord(+10, 0, 0),
    Coord(0, +10, 0), Coord(-10, 0, 0),
    Coord(0, 0, +10),
  };
  int face_n_node = 9;
  // test the version without conversion:
  {
    using Vector = std::vector<size_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66, 77, 88, 99, 1010,
      1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020,
      2121, 2222, 2323, 2424, 2525, 2626, 00 };
    Vector face_nodes, face_nodes_expect;
    face_nodes = { 11, 22, 33, 44, 2121, 1111, 1010, 1212, 99, 0 };
    face_nodes_expect = { 11, 44, 33, 22, 1212, 1111, 1010, 99, 2121, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 22, 55, 66, 2222, 1313, 1414, 1717, 99, 0 };
    face_nodes_expect = { 11, 22, 66, 55, 99, 1414, 1717, 1313, 2222, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 44, 55, 88, 2525, 1212, 1313, 2020, 1616, 0 };
    face_nodes_expect = { 11, 55, 88, 44, 1313, 2020, 1616, 1212, 2525, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 22, 33, 66, 77, 2323, 1818, 1515, 1414, 1010, 0 };
    face_nodes_expect = { 22, 33, 77, 66, 1010, 1515, 1818, 1414, 2323, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 33, 44, 77, 88, 2424, 1919, 1616, 1515, 1111, 0 };
    face_nodes_expect = { 33, 44, 88, 77, 1111, 1616, 1919, 1515, 2424, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 66, 77, 88, 2626, 2020, 1919, 1818, 1717, 0 };
    face_nodes_expect = { 55, 66, 77, 88, 1717, 1818, 1919, 2020, 2626, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
  // test the version with conversion:
  {
    using Vector = std::vector<int16_t>;
    Vector cell_nodes{ 11, 22, 33, 44, 55, 66, 77, 88, 99, 1010,
      1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020,
      2121, 2222, 2323, 2424, 2525, 2626, 00 };
    Vector face_nodes, face_nodes_expect;
    face_nodes = { 11, 22, 33, 44, 2121, 1111, 1010, 1212, 99, 0 };
    face_nodes_expect = { 11, 44, 33, 22, 1212, 1111, 1010, 99, 2121, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 22, 55, 66, 2222, 1313, 1414, 1717, 99, 0 };
    face_nodes_expect = { 11, 22, 66, 55, 99, 1414, 1717, 1313, 2222, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 11, 44, 55, 88, 2525, 1212, 1313, 2020, 1616, 0 };
    face_nodes_expect = { 11, 55, 88, 44, 1313, 2020, 1616, 1212, 2525, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 22, 33, 66, 77, 2323, 1818, 1515, 1414, 1010, 0 };
    face_nodes_expect = { 22, 33, 77, 66, 1010, 1515, 1818, 1414, 2323, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 33, 44, 77, 88, 2424, 1919, 1616, 1515, 1111, 0 };
    face_nodes_expect = { 33, 44, 88, 77, 1111, 1616, 1919, 1515, 2424, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
    face_nodes = { 55, 66, 77, 88, 2626, 2020, 1919, 1818, 1717, 0 };
    face_nodes_expect = { 55, 66, 77, 88, 1717, 1818, 1919, 2020, 2626, 0 };
    SortNodesOnFace(cell, cell_nodes.data(), face_nodes.data(), face_n_node);
    EXPECT_EQ(face_nodes, face_nodes_expect);
  }
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
