// Copyright 2021 PEI Weicheng and YANG Minghao and JIANG Yuyan

#include <cassert>
#include <cstdio>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "mini/mesh/metis/format.hpp"
#include "mini/mesh/metis/partitioner.hpp"
#include "mini/data/path.hpp"  // defines PROJECT_BINARY_DIR

namespace mini {
namespace mesh {
namespace metis {

class Partitioner : public ::testing::Test {
 protected:
  std::string const project_binary_dir_{PROJECT_BINARY_DIR};
  using Int = idx_t;
  using Real = real_t;
  std::vector<Int> null_vector_of_idx;
  std::vector<Real> null_vector_of_real;
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
  auto* file = fopen(name, "w");
  Int n_cells = n_cells_x * n_cells_y;
  Int n_nodes_x{n_cells_x + 1}, n_nodes_y{n_cells_y + 1};
  Int n_nodes = n_nodes_x * n_nodes_y;
  fprintf(file, "# vtk DataFile Version 2.0\n");  // Version and Identifier
  fprintf(file, "An unstructed mesh partitioned by METIS.\n");  // Header
  fprintf(file, "ASCII\n");  // File Format
  // DataSet Structure
  fprintf(file, "DATASET UNSTRUCTURED_GRID\n");
  fprintf(file, "POINTS %d float\n", n_nodes);
  for (Int j = 0; j != n_nodes_y; ++j) {
    for (Int i = 0; i != n_nodes_x; ++i) {
      fprintf(file, "%f %f 0.0\n", static_cast<float>(i),
                                   static_cast<float>(j));
    }
  }
  fprintf(file, "CELLS %d %d\n", n_cells, n_cells * 5);
  auto* curr_node = &(mesh.nodes(0));
  for (int i = 0; i < mesh.CountNodes(); ++i) {
    fprintf(file, "4 %d %d %d %d\n",
            curr_node[0], curr_node[1], curr_node[2], curr_node[3]);
    curr_node += 4;
  }
  fprintf(file, "CELL_TYPES %d\n", n_cells);
  for (Int i = 0; i != n_cells; ++i) {
    fprintf(file, "9\n");  // VTK_QUAD = 9
  }
  fprintf(file, "CELL_DATA %d\n", n_cells);
  fprintf(file, "SCALARS CellPartID float 1\n");
  fprintf(file, "LOOKUP_TABLE cell_parts\n");
  for (auto x : cell_parts) {
    fprintf(file, "%f\n", static_cast<float>(x));
  }
  fprintf(file, "SCALARS CellWeight float 1\n");
  fprintf(file, "LOOKUP_TABLE cell_weights\n");
  for (auto x : cell_weights) {
    fprintf(file, "%f\n", static_cast<float>(x));
  }
  if (node_parts.size()) {
    fprintf(file, "POINT_DATA %d\n", n_nodes);
    fprintf(file, "SCALARS NodePartID float 1\n");
    fprintf(file, "LOOKUP_TABLE node_parts\n");
    for (auto x : node_parts) {
      fprintf(file, "%f\n", static_cast<float>(x));
    }
  }
  fclose(file);
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
  Int n_parts{8}, n_common_nodes{2}, edge_cut{0};
  // Int options[METIS_NOPTIONS];
  // options[METIS_OPTION_NUMBERING] = 0;
  std::vector<Int> cell_parts, node_parts;
  PartMesh(mesh,
      cell_weights/* computational cost */,
      null_vector_of_idx/* communication size */,
      n_common_nodes, n_parts,
      null_vector_of_real/* weight of each part */,
      null_vector_of_idx/* options */,
      &edge_cut, &cell_parts, &node_parts);
  // Write the partitioned mesh:
  auto output = project_binary_dir_ + "/test/mesh/partitioned_mesh.vtk";
  WritePartitionedMesh(output.c_str(), n_cells_x, n_cells_y,
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
  auto graph = MeshToDual(n_cells, n_nodes, mesh, n_common_nodes, index_base);
  // Partition the mesh:
  auto cell_weights = std::vector<Int>(n_cells, 1);
  for (Int j = 0; j != n_cells_y; ++j) {
    for (Int i = 0; i != n_cells_x/4; ++i) {
      cell_weights[i + j * n_cells_x] = 4;
    }
  }
  Int n_constraints{1}, n_parts{8}, edge_cut{0};
  // Int options[METIS_NOPTIONS];
  // options[METIS_OPTION_NUMBERING] = 0;
  std::vector<Int> cell_parts;
  PartGraphKway(
      n_constraints, graph,
      cell_weights, null_vector_of_idx/* communication size */,
      null_vector_of_idx/* weight of each edge (in dual graph) */,
      n_parts, null_vector_of_real/* weight of each part */,
      null_vector_of_real/* unbalance tolerance */,
      null_vector_of_idx/* options */, &edge_cut, &cell_parts);
  // Write the partitioned mesh:
  auto output = project_binary_dir_ + "/test/mesh/partitioned_dual_graph.vtk";
  WritePartitionedMesh(output.c_str(), n_cells_x, n_cells_y,
      mesh, cell_weights, cell_parts, {}/* node_parts */);
}

}  // namespace metis
}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
