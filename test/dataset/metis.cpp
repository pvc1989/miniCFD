// Copyright 2021 PEI Weicheng and YANG Minghao and JIANG Yuyan

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "mini/dataset/metis.hpp"

namespace mini {
namespace mesh {
namespace metis {

class Partitioner : public ::testing::Test {
 protected:
  using Int = idx_t;
  using Real = real_t;
  static Mesh<Int> BuildSimpleMesh(Int n_cells_x, Int n_cells_y);
  static void WritePartitionedMesh(
      const char* name, Int n_cells_x, Int n_cells_y,
      const Mesh<Int> &mesh, const std::vector<Int> &cell_weights,
      const std::vector<Int> &cell_parts, const std::vector<Int> &node_parts);
};
Mesh<Partitioner::Int> Partitioner::BuildSimpleMesh(
    Int n_cells_x, Int n_cells_y) {
  Mesh<Int> mesh;
  Int n_cells = n_cells_x * n_cells_y;
  Int n_nodes_x{n_cells_x + 1}, n_nodes_y{n_cells_y + 1};
  Int n_nodes = n_nodes_x * n_nodes_y;
  mesh.resize(n_cells, n_nodes, n_cells * 4);
  Int* curr_node = &(mesh.nodes(0));
  Int* curr_cell = &(mesh.range(0));
  *curr_cell = 0;
  for (Int j = 0; j != n_cells_y; ++j) {
    for (Int i = 0; i != n_cells_x; ++i) {
      auto next = 4 + *curr_cell;
      *++curr_cell = next;
      auto k_node = i + j * n_nodes_x;  // node index starts from 0 in C code
      *curr_node++ = k_node++;
      *curr_node++ = k_node;
      k_node += n_nodes_x;
      *curr_node++ = k_node--;
      *curr_node++ = k_node;
    }
  }
  return mesh;
}
void Partitioner::WritePartitionedMesh(
    const char* name, Int n_cells_x, Int n_cells_y,
    const Mesh<Int> &mesh, const std::vector<Int> &cell_weights,
    const std::vector<Int> &cell_parts, const std::vector<Int> &node_parts) {
  auto ostrm = std::ofstream(name);
  Int n_cells = n_cells_x * n_cells_y;
  Int n_nodes_x{n_cells_x + 1}, n_nodes_y{n_cells_y + 1};
  Int n_nodes = n_nodes_x * n_nodes_y;
  ostrm << "# vtk DataFile Version 2.0\n";  // Version and Identifier
  ostrm << "An unstructed mesh partitioned by METIS.\n";  // Header
  ostrm << "ASCII\n";  // File Format
  // DataSet Structure
  ostrm << "DATASET UNSTRUCTURED_GRID\n";
  ostrm << "POINTS " << n_nodes << " float\n";
  for (Int j = 0; j != n_nodes_y; ++j) {
    for (Int i = 0; i != n_nodes_x; ++i) {
      ostrm << static_cast<float>(i) << " "
            << static_cast<float>(j) << " 0.0\n";
    }
  }
  ostrm << "CELLS " << n_cells << " " << n_cells * 5 << "\n";
  auto *curr_node = &(mesh.nodes(0));
  for (int i = 0; i < mesh.CountNodes(); ++i) {
    ostrm << "4 " << curr_node[0] << " " << curr_node[1] << " "
                  << curr_node[2] << " " << curr_node[3] << "\n";
    curr_node += 4;
  }
  ostrm << "CELL_TYPES " << n_cells << "\n";
  for (Int i = 0; i != n_cells; ++i) {
    ostrm << "9\n";  // VTK_QUAD = 9
  }
  ostrm << "CELL_DATA " << n_cells << "\n";
  ostrm << "SCALARS PartIndex float 1\n";
  ostrm << "LOOKUP_TABLE cell_parts\n";
  for (auto x : cell_parts) {
    ostrm << static_cast<float>(x) << "\n";
  }
  ostrm << "SCALARS CellWeight float 1\n";
  ostrm << "LOOKUP_TABLE cell_weights\n";
  for (auto x : cell_weights) {
    ostrm << static_cast<float>(x) << "\n";
  }
  if (node_parts.size()) {
    ostrm << "POINT_DATA " << n_nodes << "\n";
    ostrm << "SCALARS PartIndex float 1\n";
    ostrm << "LOOKUP_TABLE node_parts\n";
    for (auto x : node_parts) {
      ostrm << static_cast<float>(x) << "\n";
    }
  }
}
TEST_F(Partitioner, PartMesh) {
  // Build a simple mesh:
  Int n_cells_x{100}, n_cells_y{40};
  Int n_cells = n_cells_x * n_cells_y;
  Int n_nodes_x{n_cells_x + 1}, n_nodes_y{n_cells_y + 1};
  Int n_nodes = n_nodes_x * n_nodes_y;
  auto mesh = BuildSimpleMesh(n_cells_x, n_cells_y);
  EXPECT_EQ(mesh.CountCells(), n_cells);
  EXPECT_EQ(mesh.CountNodes(), n_nodes);
  // Partition the mesh:
  auto cell_weights = std::vector<Int>(n_cells, 1);
  for (Int j = 0; j != n_cells_y; ++j) {
    for (Int i = 0; i != n_cells_x/4; ++i) {
      cell_weights[i + j * n_cells_x] = 4;
    }
  }
  Int n_parts{8}, n_common_nodes{2};
  // Int options[METIS_NOPTIONS];
  // options[METIS_OPTION_NUMBERING] = 0;
  auto [cell_parts, node_parts] = PartMesh(
      mesh, n_parts, n_common_nodes, cell_weights);
  // Write the partitioned mesh:
  WritePartitionedMesh("partitioned_mesh.vtk", n_cells_x, n_cells_y,
      mesh, cell_weights, cell_parts, node_parts);
}
TEST_F(Partitioner, PartGraphKway) {
  // Build a simple mesh:
  Int n_cells_x{100}, n_cells_y{40};
  Int n_cells = n_cells_x * n_cells_y;
  Int n_nodes_x{n_cells_x + 1}, n_nodes_y{n_cells_y + 1};
  Int n_nodes = n_nodes_x * n_nodes_y;
  auto mesh = BuildSimpleMesh(n_cells_x, n_cells_y);
  // Build the dual graph:
  Int n_common_nodes{2}, index_base{0};
  auto graph = MeshToDual(mesh, n_common_nodes, index_base);
  // Partition the mesh:
  auto cell_weights = std::vector<Int>(n_cells, 1);
  for (Int j = 0; j != n_cells_y; ++j) {
    for (Int i = 0; i != n_cells_x/4; ++i) {
      cell_weights[i + j * n_cells_x] = 4;
    }
  }
  Int n_constraints{1}, n_parts{8};
  // Int options[METIS_NOPTIONS];
  // options[METIS_OPTION_NUMBERING] = 0;
  auto cell_parts = PartGraph(
      graph, n_parts, n_constraints, cell_weights);
  // Write the partitioned mesh:
  WritePartitionedMesh("partitioned_dual_graph.vtk", n_cells_x, n_cells_y,
      mesh, cell_weights, cell_parts, {}/* node_parts */);
}
TEST_F(Partitioner, GetNodeParts) {
  std::vector<int> cell_parts{2, 0, 0, 1};
  int n_nodes = 10;
  auto mesh = Mesh<int>(
      {0, 4, 8, 11, 14},
      {0, 2, 3, 1,   2, 4, 5, 3,   6, 8, 7,   8, 9, 7},
      n_nodes);
  int n_parts = 3;
  auto node_parts = GetNodeParts(mesh, cell_parts, n_parts);
  std::vector<int> expected_node_parts{2, 2, 0, 0, 0, 0, 0, 0, 0, 1};
  for (int i = 0; i < n_nodes; ++i) {
    EXPECT_EQ(node_parts[i], expected_node_parts[i]);
  }
}

}  // namespace metis
}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
