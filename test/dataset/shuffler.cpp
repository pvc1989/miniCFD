// Copyright 2019 Weicheng Pei and Minghao Yang

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "mini/mesh/mapper.hpp"
#include "mini/mesh/shuffler.hpp"
#include "mini/mesh/cgns.hpp"
#include "mini/mesh/metis.hpp"
#include "mini/data/path.hpp"  // defines TEST_DATA_DIR

namespace mini {
namespace mesh {

idx_t n_parts = 4;
char case_name[33] = "double_mach";

class ShufflerTest : public ::testing::Test {
 protected:
  using Shuffler = mesh::Shuffler<idx_t, double>;
  std::string const test_data_dir_{TEST_DATA_DIR};
};
TEST_F(ShufflerTest, GetNewOrder) {
  // Reorder the indices by parts
  int n = 10;
  std::vector<int> old_index{2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<int>     parts{1, 0, 0, 2, 0, 1, 2, 1,  0,  2};
  auto [new_to_old, old_to_new] = GetNewOrder(parts.data(), parts.size());
  std::vector<int> expected_new_to_old{1, 2, 4, 8, 0, 5, 7, 3, 6, 9};
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(new_to_old[i], expected_new_to_old[i]);
  }
  // Shuffle data array by new order
  std::vector<double> old_array{
      1.0, 0.0, 0.1, 2.0, 0.2, 1.1, 2.1, 1.2, 0.3, 2.2};
  ShuffleData(new_to_old, old_array.data());
  std::vector<double> expected_new_array{
      0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2};
  for (int i = 0; i < n; ++i) {
    EXPECT_DOUBLE_EQ(old_array[i], expected_new_array[i]);
  }
}
TEST_F(ShufflerTest, ShuffleConnectivity) {
  // Shuffle cell i_node_list by new order
  std::vector<int> new_to_old_for_nodes{5, 4, 3, 2, 1, 0};
  std::vector<int> new_to_old_for_cells{1, 2, 3, 0};
  std::vector<int> i_node_list = {
      1, 2, 6,   2, 3, 6,   6, 3, 5,   3, 4, 5};
  int npe = 3;
  ShuffleConnectivity(
      new_to_old_for_nodes, new_to_old_for_cells, npe, i_node_list.data());
  std::vector<int> expected_new_i_node_list{
      5, 4, 1,   1, 4, 2,   4, 3, 2,   6, 5, 1};
  for (int i = 0; i < i_node_list.size(); ++i) {
    EXPECT_EQ(i_node_list[i], expected_new_i_node_list[i]);
  }
}
TEST_F(ShufflerTest, ParitionAndShuffle) {
  char cmd[1024];
  std::snprintf(cmd, sizeof(cmd), "mkdir -p %s/partition", case_name);
  std::system(cmd); std::cout << "[Done] " << cmd << std::endl;
  auto old_file_name = case_name + std::string("/original.cgns");
  /* Generate the original cgns file: */
  std::snprintf(cmd, sizeof(cmd), "gmsh %s/%s.geo -save -o %s",
      test_data_dir_.c_str(), case_name, old_file_name.c_str());
  std::system(cmd); std::cout << "[Done] " << cmd << std::endl;
  Shuffler::PartitionAndShuffle(case_name, old_file_name, n_parts);
}

}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  if (argc > 1) {
    mini::mesh::n_parts = std::atoi(argv[1]);
  }
  if (argc > 2) {
    std::strncpy(mini::mesh::case_name, argv[2], sizeof(mini::mesh::case_name));
  }
  return RUN_ALL_TESTS();
}
