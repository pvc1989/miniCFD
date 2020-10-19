// Copyright 2019 Weicheng Pei and Minghao Yang

#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "cgnslib.h"
#include "metis.h"
#include "gtest/gtest.h"

#include "mini/mesh/cgns/converter.hpp"
#include "mini/mesh/cgns/reader.hpp"
#include "mini/mesh/cgns/shuffler.hpp"
#include "mini/mesh/cgns/tree.hpp"
#include "mini/data/path.hpp"  // defines PROJECT_BINARY_DIR

namespace mini {
namespace mesh {
namespace metis{

class Shuffler : public ::testing::Test {
};

TEST_F(Shuffler, ShuffleByParts) {
  // Select local indices from global indices.
  int n = 5;
  std::vector<int> global_cell_parts{0, 1, 2, 0, 1, 2, 1, 0, 1, 0};
  std::vector<int> selected_cell_indices{0, 2, 4, 5, 8};
  std::vector<int> local_cell_parts;
  SelectByIndices<int>(global_cell_parts, selected_cell_indices, local_cell_parts);
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(local_cell_parts[i], global_cell_parts[selected_cell_indices[i]]);
  }
  // Reorder the indices by parts
  n = 10;
  std::vector<int> old_index{2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<int>     parts{1, 0, 0, 2, 0, 1, 2, 1,  0,  2};
  std::vector<int> new_order;
  ReorderByParts<int>(parts, new_order);
  std::vector<int> expected_new_order{1, 2, 4, 8, 0, 5, 7, 3, 6, 9};
  for (int i = 0; i < n; ++ i) {
    EXPECT_EQ(new_order[i], expected_new_order[i]);
  }
  // Shuffle data array by new order
  std::vector<double> old_array{1.0, 0.0, 0.1, 2.0, 0.2, 1.1, 2.1, 1.2, 0.3, 2.2};
  std::vector<double> new_array;
  ShuffleDataArray<double, int>(old_array, new_order, new_array);
  std::vector<double> expected_new_array{0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2};
  for (int i = 0; i < n; ++i) {
    EXPECT_DOUBLE_EQ(new_array[i], expected_new_array[i]);
  }
  // Shuffle cell connectivity by new order
  n = 4;
  std::vector<int> cell_parts{2, 0, 0, 1};
  std::vector<int> node_parts{2, 0, 0, 1, 1, 0};
  std::vector<int> new_cell_order{1, 2, 3, 0};
  std::vector<int> old_to_new_for_node{0, 2, 3, 6, 4, 5, 1};
  std::vector<int> old_connectivity = {1, 2, 6,   2, 3, 6,   6, 3, 5,   3, 4 ,5};
  std::vector<int> new_connectivity;
  ShuffleConnectivity<int>(old_to_new_for_node, new_cell_order,
                           old_connectivity, new_connectivity);
  std::vector<int> expected_new_connectivity{3, 6, 1,   1, 6, 5,   6, 4, 5,
                                             2, 3, 1};
  for (int i = 0; i < new_connectivity.size(); ++i) {
    EXPECT_DOUBLE_EQ(new_connectivity[i], expected_new_connectivity[i]);
  }
}

class Partition : public ::testing::Test {
  protected:
   using CSRM = mini::mesh::cgns::CompressedSparseRowMatrix;
   CSRM cell_csrm;
};

TEST_F(Partition, GetNodePartsByConnectivity) {
  std::vector<int> cell_parts{2, 0, 0, 1};
  cell_csrm.pointer = {0, 4, 8, 11, 14};
  cell_csrm.index = {0, 2, 3, 1,   2, 4, 5, 3,   6, 8, 7,   8, 9, 7};
  int n_nodes = 10;
  int n_parts = 3;
  std::vector<int> node_parts;
  GetNodePartsByConnectivity<int>(cell_csrm, cell_parts, n_parts, n_nodes, node_parts);
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