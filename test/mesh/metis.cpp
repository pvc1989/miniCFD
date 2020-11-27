// Copyright 2019 Weicheng Pei and Minghao Yang

#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "metis.h"
#include "gtest/gtest.h"
#include "mini/mesh/metis/partitioner.hpp"
#include "mini/data/path.hpp"  // defines PROJECT_BINARY_DIR

namespace mini {
namespace mesh {
namespace metis {

class Partitioner : public ::testing::Test {
 protected:
  std::string const project_binary_dir_{PROJECT_BINARY_DIR};
  template <typename Int>
  static void BuildSimpleMesh(Int n_cells_x, Int n_cells_y,
      std::vector<Int> *cell_nodes, std::vector<Int> *cell_range);
};
template <typename Int>
void Partitioner::BuildSimpleMesh(Int n_cells_x, Int n_cells_y,
    std::vector<Int> *cell_nodes, std::vector<Int> *cell_range) {
  Int n_cells = n_cells_x * n_cells_y;
  Int n_nodes_x{n_cells_x + 1}, n_nodes_y{n_cells_y + 1};
  Int n_nodes = n_nodes_x * n_nodes_y;
  cell_nodes->resize(n_cells * 4);  // indices of nodes in each cell
  cell_range->resize(n_cells + 1);  // use to slice cell_nodes
  auto curr_node = cell_nodes->begin();
  auto curr_cell = cell_range->begin();
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
  ++curr_cell;
  assert(curr_node == cell_nodes->end());
  assert(curr_cell == cell_range->end());
}
TEST_F(Partitioner, PartMeshDual) {
  /*
    Partition a mesh into k parts based on a partitioning of its dual graph.
   */
  using Int = int;
  using Real = float;
  // Build a simple mesh:
  Int n_cells_x{100}, n_cells_y{40};
  Int n_cells = n_cells_x * n_cells_y;
  Int n_nodes_x{n_cells_x + 1}, n_nodes_y{n_cells_y + 1};
  Int n_nodes = n_nodes_x * n_nodes_y;
  std::vector<Int> cell_nodes;  // indices of nodes in each cell
  std::vector<Int> cell_range;  // use to slice cell_nodes
  BuildSimpleMesh(n_cells_x, n_cells_y, &cell_nodes, &cell_range);
  EXPECT_EQ(cell_nodes.size(), n_cells * 4);
  EXPECT_EQ(cell_range.size(), n_cells + 1);
  // Partition the mesh:
  auto cell_weights = std::vector<Int>(n_cells, 1);
  for (Int j = 0; j != n_cells_y; ++j) {
    for (Int i = 0; i != n_cells_x/4; ++i) {
      cell_weights[i + j * n_cells_x] = 4;
    }
  }
  auto cell_parts = std::vector<Int>(n_cells);
  auto node_parts = std::vector<Int>(n_nodes);
  Int n_parts{8}, n_common_nodes{2}, edge_cut{0};
  // Int options[METIS_NOPTIONS];
  // options[METIS_OPTION_NUMBERING] = 0;
  auto null_vector_of_idx = std::vector<Int>();
  auto null_vector_of_real = std::vector<Real>();
  auto result = PartMeshDual(
      n_cells, n_nodes, cell_range, cell_nodes,
      cell_weights/* computational cost */,
      null_vector_of_idx/* communication size */,
      n_common_nodes, n_parts,
      null_vector_of_real/* weight of each part */,
      null_vector_of_idx/* options */,
      &edge_cut, &cell_parts, &node_parts);
  EXPECT_EQ(result, METIS_OK);
  // Print the mesh:
  /*
  std::cout << n_cells << std::endl;
  curr_node = cell_nodes.begin();
  for (Int i = 0; i != n_cells; ++i) {
    for (Int j = 0; j != 4; ++j) {
      std::cout << 1 + *curr_node++ // node index starts from 1 in mesh file
                << ' ';
    }
    std::cout << std::endl;
  }
  EXPECT_EQ(curr_node, cell_nodes.end());
   */
  // Print the result:
  /*
  std::cout << "part id of each cell: ";
  for (auto x : cell_parts) {
    std::cout << x << ' ';
  }
  std::cout << std::endl;
  std::cout << "part id of each node: ";
  for (auto x : node_parts) {
    std::cout << x << ' ';
  }
  std::cout << std::endl;
   */
  // Write partitioned mesh:
  auto output = project_binary_dir_ + "/test/mesh/partitioned_mesh.vtk";
  auto* file = fopen(output.c_str(), "w");
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
  auto curr_node = cell_nodes.begin();
  while (curr_node != cell_nodes.end()) {
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
  fprintf(file, "POINT_DATA %d\n", n_nodes);
  fprintf(file, "SCALARS NodePartID float 1\n");
  fprintf(file, "LOOKUP_TABLE node_parts\n");
  for (auto x : node_parts) {
    fprintf(file, "%f\n", static_cast<float>(x));
  }
  fclose(file);
}
TEST_F(Partitioner, PartGraphKway) {
  /*
    Partition a mesh's dual graph into k parts.
   */
  using Int = int;
  using Real = float;
  // Build a simple mesh:
  Int n_cells_x{100}, n_cells_y{40};
  Int n_cells = n_cells_x * n_cells_y;
  Int n_nodes_x{n_cells_x + 1}, n_nodes_y{n_cells_y + 1};
  Int n_nodes = n_nodes_x * n_nodes_y;
  std::vector<Int> cell_nodes;  // indices of nodes in each cell
  std::vector<Int> cell_range;  // use to slice cell_nodes
  BuildSimpleMesh(n_cells_x, n_cells_y, &cell_nodes, &cell_range);
  EXPECT_EQ(cell_nodes.size(), n_cells * 4);
  EXPECT_EQ(cell_range.size(), n_cells + 1);
  // Build the dual graph:
  Int n_common_nodes{2}, index_base{0};
  Int *range_of_each_cell, *neighbors_of_each_cell;
  auto result = METIS_MeshToDual(
    &n_cells, &n_nodes, cell_range.data(), cell_nodes.data(),
    &n_common_nodes, &index_base,
    &range_of_each_cell, &neighbors_of_each_cell);
  EXPECT_EQ(result, METIS_OK);
  // Partition the mesh:
  auto cell_weights = std::vector<Int>(n_cells, 1);
  for (Int j = 0; j != n_cells_y; ++j) {
    for (Int i = 0; i != n_cells_x/4; ++i) {
      cell_weights[i + j * n_cells_x] = 4;
    }
  }
  auto cell_parts = std::vector<Int>(n_cells);
  Int n_constraints{1}, n_parts{8}, edge_cut{0};
  // Int options[METIS_NOPTIONS];
  // options[METIS_OPTION_NUMBERING] = 0;
  result = METIS_PartGraphKway(
    &n_cells, &n_constraints, range_of_each_cell, neighbors_of_each_cell,
    cell_weights.data()/* computational cost */,
    NULL/* communication size */, NULL/* weight of each edge (in dual graph) */,
    &n_parts, NULL/* weight of each part */, NULL/* unbalance tolerance */,
    NULL/* options */, &edge_cut, cell_parts.data());
  EXPECT_EQ(result, METIS_OK);
  METIS_Free(range_of_each_cell);
  METIS_Free(neighbors_of_each_cell);
  // Write partitioned mesh:
  auto output = project_binary_dir_ + "/test/mesh/partitioned_dual_graph.vtk";
  auto* file = fopen(output.c_str(), "w");
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
  auto curr_node = cell_nodes.begin();
  while (curr_node != cell_nodes.end()) {
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
  fclose(file);
}

}  // namespace metis
}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
