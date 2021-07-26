// Copyright 2019 Weicheng Pei and Minghao Yang

#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "metis.h"
#include "gtest/gtest.h"

#include "mini/data/path.hpp"  // defines PROJECT_BINARY_DIR

namespace mini {
namespace mesh {
namespace metis {

class Partitioner : public ::testing::Test {
 protected:
  std::string const project_binary_dir_{PROJECT_BINARY_DIR};
  static void BuildSimpleMesh(idx_t n_elems_x, idx_t n_elems_y,
      std::vector<idx_t> *elem_nodes, std::vector<idx_t> *elem_range);
};
void Partitioner::BuildSimpleMesh(idx_t n_elems_x, idx_t n_elems_y,
    std::vector<idx_t> *elem_nodes, std::vector<idx_t> *elem_range) {
  idx_t n_elems = n_elems_x * n_elems_y;
  idx_t n_nodes_x{n_elems_x + 1}, n_nodes_y{n_elems_y + 1};
  idx_t n_nodes = n_nodes_x * n_nodes_y;
  elem_nodes->resize(n_elems * 4);  // indices of nodes in each elem
  elem_range->resize(n_elems + 1);  // use to slice elem_nodes
  auto curr_node = elem_nodes->begin();
  auto curr_elem = elem_range->begin();
  *curr_elem = 0;
  for (int j = 0; j != n_elems_y; ++j) {
    for (int i = 0; i != n_elems_x; ++i) {
      auto next = 4 + *curr_elem;
      *++curr_elem = next;
      auto k_node = i + j * n_nodes_x;  // node index starts from 0 in C code
      *curr_node++ = k_node++;
      *curr_node++ = k_node;
      k_node += n_nodes_x;
      *curr_node++ = k_node--;
      *curr_node++ = k_node;
    }
  }
  ++curr_elem;
  assert(curr_node == elem_nodes->end());
  assert(curr_elem == elem_range->end());
}
TEST_F(Partitioner, PartMeshDual) {
  /*
    Partition a mesh into k parts based on a partitioning of its dual graph.
   */
  // Build a simple mesh:
  idx_t n_elems_x{100}, n_elems_y{40};
  idx_t n_elems = n_elems_x * n_elems_y;
  idx_t n_nodes_x{n_elems_x + 1}, n_nodes_y{n_elems_y + 1};
  idx_t n_nodes = n_nodes_x * n_nodes_y;
  std::vector<idx_t> elem_nodes;  // indices of nodes in each elem
  std::vector<idx_t> elem_range;  // use to slice elem_nodes
  BuildSimpleMesh(n_elems_x, n_elems_y, &elem_nodes, &elem_range);
  EXPECT_EQ(elem_nodes.size(), n_elems * 4);
  EXPECT_EQ(elem_range.size(), n_elems + 1);
  // Partition the mesh:
  auto elem_weights = std::vector<idx_t>(n_elems, 1);
  for (int j = 0; j != n_elems_y; ++j) {
    for (int i = 0; i != n_elems_x/4; ++i) {
      elem_weights[i + j * n_elems_x] = 4;
    }
  }
  auto elem_parts = std::vector<idx_t>(n_elems);
  auto node_parts = std::vector<idx_t>(n_nodes);
  idx_t n_parts{8}, n_common_nodes{2}, edge_cut{0};
  // idx_t options[METIS_NOPTIONS];
  // options[METIS_OPTION_NUMBERING] = 0;
  auto result = METIS_PartMeshDual(
    &n_elems, &n_nodes, elem_range.data(), elem_nodes.data(), 
    elem_weights.data(),
    NULL/* idx_t t *vsize */,
    &n_common_nodes, &n_parts,
    NULL/* real t *tpwgts */,
    NULL/* options */,
    &edge_cut, elem_parts.data(), node_parts.data()
  );
  EXPECT_EQ(result, METIS_OK);
  // Print the mesh:
  /*
  std::cout << n_elems << std::endl;
  curr_node = elem_nodes.begin();
  for (int i = 0; i != n_elems; ++i) {
    for (int j = 0; j != 4; ++j) {
      std::cout << 1 + *curr_node++ // node index starts from 1 in mesh file
                << ' ';
    }
    std::cout << std::endl;
  }
  EXPECT_EQ(curr_node, elem_nodes.end());
   */
  // Print the result:
  /*
  std::cout << "part id of each elem: ";
  for (auto x : elem_parts) {
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
  for (int j = 0; j != n_nodes_y; ++j) {
    for (int i = 0; i != n_nodes_x; ++i) {
      fprintf(file, "%f %f 0.0\n", (float) i, (float) j);
    }
  }
  fprintf(file, "CELLS %d %d\n", n_elems, n_elems * 5);
  auto curr_node = elem_nodes.begin();
  while (curr_node != elem_nodes.end()) {
    fprintf(file, "4 %d %d %d %d\n",
            curr_node[0], curr_node[1], curr_node[2], curr_node[3]);
    curr_node += 4;
  }
  fprintf(file, "CELL_TYPES %d\n", n_elems);
  for (int i = 0; i != n_elems; ++i) {
    fprintf(file, "9\n");  // VTK_QUAD = 9
  }
  fprintf(file, "CELL_DATA %d\n", n_elems);
  fprintf(file, "SCALARS CellPartID float 1\n");
  fprintf(file, "LOOKUP_TABLE elem_parts\n");
  for (auto x : elem_parts) {
    fprintf(file, "%f\n", (float) x);
  }
  fprintf(file, "SCALARS CellWeight float 1\n");
  fprintf(file, "LOOKUP_TABLE elem_weights\n");
  for (auto x : elem_weights) {
    fprintf(file, "%f\n", (float) x);
  }
  fprintf(file, "POINT_DATA %d\n", n_nodes);
  fprintf(file, "SCALARS NodePartID float 1\n");
  fprintf(file, "LOOKUP_TABLE node_parts\n");
  for (auto x : node_parts) {
    fprintf(file, "%f\n", (float) x);
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
