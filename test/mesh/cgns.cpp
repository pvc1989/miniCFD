// Copyright 2019 Weicheng Pei and Minghao Yang

#include <cstdio>
#include <string>
#include <vector>

#include "cgnslib.h"
#include "gtest/gtest.h"

#include "mini/mesh/cgns/reader.hpp"
#include "mini/data/path.hpp"  // defines TEST_DATA_DIR

namespace mini {
namespace mesh {
namespace cgns {

class ReaderTest : public ::testing::Test {
 protected:
  std::string const test_data_dir_{TEST_DATA_DIR};
};
TEST_F(ReaderTest, ReadFromFile) {
  int file_id;
  auto file_name = test_data_dir_ + "ugrid_2d.cgns";
  if (cg_open(file_name.c_str(), CG_MODE_READ, &file_id)) {
    cg_error_exit();
  }
  else {
    int n_bases, n_zones;
    cg_nbases(file_id, &n_bases);
    if (n_bases) {
      cg_nzones(file_id, n_bases, &n_zones);
      std::printf("There are %d `Base`s and %d `Zone`s.\n", n_bases, n_zones);
    }
    cg_close(file_id);
  }
}

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
