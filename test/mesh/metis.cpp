// Copyright 2019 Weicheng Pei and Minghao Yang

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "metis.h"
#include "gtest/gtest.h"

#include "mini/data/path.hpp"  // defines TEST_DATA_DIR

namespace mini {
namespace mesh {
namespace metis {

class Partitioner : public ::testing::Test {
 protected:
  std::string const test_data_dir_{TEST_DATA_DIR};
};
TEST_F(Partitioner, PartMeshDual) {
  /*
    Partition a mesh into k parts based on a partitioning of its dual graph.
   */
  // Build a simple mesh:
  idx_t n_elems_x{30}, n_elems_y{1};
  idx_t n_elems = n_elems_x * n_elems_y;
  idx_t n_nodes_x{n_elems_x + 1}, n_nodes_y{n_elems_y + 1};
  idx_t n_nodes = n_nodes_x * n_nodes_y;
  std::vector<idx_t> elem_nodes(n_elems * 4);  // indices of nodes in each elem
  std::vector<idx_t> elem_range(n_elems + 1);  // use to slice elem_nodes
  auto curr_node = elem_nodes.begin();
  auto curr_elem = elem_range.begin();
  *curr_elem = 0;
  for (int i = 0; i != n_elems_x; ++i) {
    for (int j = 0; j != n_elems_y; ++j) {
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
  EXPECT_EQ(curr_elem, elem_range.end());
  EXPECT_EQ(curr_node, elem_nodes.end());
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
  // Partition the mesh:
  auto elem_parts = std::vector<idx_t>(n_elems);
  auto node_parts = std::vector<idx_t>(n_nodes);
  idx_t n_parts{5}, n_common_nodes{2}, edge_cut{0};
  // idx_t options[METIS_NOPTIONS];
  // options[METIS_OPTION_NUMBERING] = 0;
  auto result = METIS_PartMeshDual(
    &n_elems, &n_nodes, elem_range.data(), elem_nodes.data(), 
    NULL/* idx_t t *vwgt */,
    NULL/* idx_t t *vsize */,
    &n_common_nodes, &n_parts,
    NULL/* real t *tpwgts */,
    NULL/* options */,
    &edge_cut, elem_parts.data(), node_parts.data()
  );
  EXPECT_EQ(result, METIS_OK);
  // Print the result:
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
}

}  // namespace metis
}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
