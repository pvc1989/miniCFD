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
  using MeshType = Tree<double>;
  Reader<MeshType> reader;
  std::string const test_data_dir_{TEST_DATA_DIR};
};
TEST_F(ReaderTest, ReadFromFile) {
  auto file_name = test_data_dir_ + "ugrid_2d.cgns";
  EXPECT_TRUE(reader.ReadFromFile(file_name));
}
TEST_F(ReaderTest, GetMesh) {
  auto file_name = test_data_dir_ + "ugrid_2d.cgns";
  EXPECT_TRUE(reader.ReadFromFile(file_name));
  auto mesh = reader.GetMesh();
  EXPECT_NE(mesh, nullptr);
}
TEST_F(ReaderTest, ReadBase) {
  auto file_name = test_data_dir_ + "ugrid_2d.cgns";
  // read by cgns
  int file_id{-1};
  cg_open(file_name.c_str(), CG_MODE_READ, &file_id);
  int n_bases{-1};
  cg_nbases(file_id, &n_bases);
  struct BaseInfo {
    std::string name; int id, cell_dim, phys_dim;
    BaseInfo(char* bn, int bi, int cd, int pd) 
      : name(bn), id(bi), cell_dim(cd), phys_dim(pd) {}
  };
  auto base_info = std::vector<BaseInfo>();
  for (int base_id = 1; base_id <= n_bases; ++base_id) {
    char base_name[33];
    int cell_dim{-1}, phys_dim{-1};
    cg_base_read(file_id, base_id, base_name, &cell_dim, &phys_dim);
    base_info.emplace_back(base_name, base_id, cell_dim, phys_dim);
  }
  cg_close(file_id);
  // read by cgns
  reader.ReadFromFile(file_name);
  auto mesh = reader.GetMesh();
  // compare result
  EXPECT_EQ(mesh->CountBases(), n_bases);
  for (auto& base : base_info) {
    auto& my_base = mesh->GetBase(base.id);
    EXPECT_STREQ(my_base.GetName().c_str(), base.name);
    EXPECT_EQ(my_base.GetCellDim(), base.cell_dim);
    EXPECT_EQ(my_base.GetPhysDim(), base.phys_dim);
  }
}

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
