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

TEST_F(Shuffler, ShuffleCellByPart) {
  int n = 10;
  std::vector<int> old_index{2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<int>     parts{1, 0, 0, 2, 0, 1, 2, 1,  0,  2};
  std::vector<int> new_order;
  int begin = old_index[0];
  new_order.reserve(n);
  ReorderByParts<int, int>(n, parts.data(), new_order.data());
  EXPECT_EQ(new_order[0], old_index[1] - begin);
  EXPECT_EQ(new_order[1], old_index[2] - begin);
  EXPECT_EQ(new_order[2], old_index[4] - begin);
  EXPECT_EQ(new_order[3], old_index[8] - begin);
  EXPECT_EQ(new_order[4], old_index[0] - begin);
  EXPECT_EQ(new_order[5], old_index[5] - begin);
  EXPECT_EQ(new_order[6], old_index[7] - begin);
  EXPECT_EQ(new_order[7], old_index[3] - begin);
  EXPECT_EQ(new_order[8], old_index[6] - begin);
  EXPECT_EQ(new_order[9], old_index[9] - begin);
  std::vector<double> old_array{1.0, 0.0, 0.1, 2.0, 0.2, 1.1, 2.1, 1.2, 0.3, 2.2};
  std::vector<double> new_array;
  new_array.reserve(n);
  ShuffleDataArray<double, int>(n, old_array.data(), new_order.data(),
                                new_array.data());
  EXPECT_DOUBLE_EQ(new_array[0], 0.0);
  EXPECT_DOUBLE_EQ(new_array[1], 0.1);
  EXPECT_DOUBLE_EQ(new_array[2], 0.2);
  EXPECT_DOUBLE_EQ(new_array[3], 0.3);
  EXPECT_DOUBLE_EQ(new_array[4], 1.0);
  EXPECT_DOUBLE_EQ(new_array[5], 1.1);
  EXPECT_DOUBLE_EQ(new_array[6], 1.2);
  EXPECT_DOUBLE_EQ(new_array[7], 2.0);
  EXPECT_DOUBLE_EQ(new_array[8], 2.1);
  EXPECT_DOUBLE_EQ(new_array[9], 2.2);
}


}  // namespace metis
}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}