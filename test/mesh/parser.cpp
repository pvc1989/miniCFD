//  Copyright 2021 PEI Weicheng and JIANG Yuyan

#include "mini/mesh/cgns/parser.hpp"

#include "gtest/gtest.h"
#include "mini/data/path.hpp"  // defines PROJECT_BINARY_DIR

namespace mini {
namespace mesh {
namespace cgns {

class TestParser : public ::testing::Test {
 protected:
  std::string const current_binary_dir_{
      std::string(PROJECT_BINARY_DIR) + std::string("/test/mesh")};
};

TEST_F(TestParser, Print) {
  std::string cgns_file = "";
  auto prefix = current_binary_dir_ + "/ugrid_2d_part_";
  int pid = 0;
  auto parser = Parser(cgns_file, prefix, pid);
}

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
