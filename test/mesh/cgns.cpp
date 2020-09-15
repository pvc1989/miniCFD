// Copyright 2019 Weicheng Pei and Minghao Yang

#include <cstdio>
#include <string>
#include <vector>

#include "cgnslib.h"
#include "gtest/gtest.h"

#include "mini/mesh/cgns/reader.hpp"
#include "mini/mesh/cgns/tree.hpp"
#include "mini/data/path.hpp"  // defines TEST_DATA_DIR

namespace mini {
namespace mesh {
namespace cgns {

class ReaderTest : public ::testing::Test {
 protected:
  using MeshType = Tree<double>;
  using Coordinates = std::vector<std::vector<double>>;
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
  // read by cgnslib
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
  // read by mini::mesh::cgns
  reader.ReadFromFile(file_name);
  auto mesh = reader.GetMesh();
  // compare result
  EXPECT_EQ(mesh->CountBases(), n_bases);
  for (auto& base : base_info) {
    auto& my_base = mesh->GetBase(base.id);
    EXPECT_STREQ(my_base.GetName().c_str(), base.name.c_str());
    EXPECT_EQ(my_base.GetCellDim(), base.cell_dim);
    EXPECT_EQ(my_base.GetPhysDim(), base.phys_dim);
  }
}

TEST_F(ReaderTest, ReadZone) {
  auto file_name = test_data_dir_ + "ugrid_2d.cgns";
  // read by cgnslib
  int file_id{-1};
  cg_open(file_name.c_str(), CG_MODE_READ, &file_id);
  struct ZoneInfo {
    std::string name; int id, vertex_size, cell_size;
    std::vector<double> x, y, z;
    ZoneInfo(char* zn, int zi, int* zone_size) 
        : name(zn), id(zi), cell_size(zone_size[1]),
          x(zone_size[0]), y(zone_size[0]), z(zone_size[0]) {}
  };
  auto zone_info = std::vector<ZoneInfo>();
  int n_bases{-1};
  cg_nbases(file_id, &n_bases);
  for (int base_id = 1; base_id <= n_bases; ++base_id) {
    int n_zones{-1};
    cg_nzones(file_id, base_id, &n_zones);
    for (int zone_id = 1; zone_id <= n_zones; ++zone_id) {
      char zone_name[33];
      int zone_size[3][1];
      cg_zone_read(file_id, base_id, zone_id, zone_name, zone_size[0]);
      auto& cg_zone = zone_info.emplace_back(zone_name, zone_id, zone_size[0]);
      int first = 0;
      int last = cg_zone.x.size();
      cg_coord_read(file_id, base_id, zone_id, "CoordinateX",
                    CGNS_ENUMV(RealSingle), &first, &last, cg_zone.x.data());
      cg_coord_read(file_id, base_id, zone_id, "CoordinateY",
                    CGNS_ENUMV(RealSingle), &first, &last, cg_zone.y.data());
      cg_coord_read(file_id, base_id, zone_id, "CoordinateZ",
                    CGNS_ENUMV(RealSingle), &first, &last, cg_zone.z.data());
    }
  }
  cg_close(file_id);
  // read by mini::mesh::cgns
  reader.ReadFromFile(file_name);
  auto mesh = reader.GetMesh();
  // compare result
  cg_open(file_name.c_str(), CG_MODE_READ, &file_id);
  EXPECT_EQ(mesh->CountBases(), n_bases);
  int index = 0;
  for (int base_id = 1; base_id <= n_bases; ++base_id) {
    auto& my_base = mesh->GetBase(base_id);
    int n_zones{0};
    cg_nzones(file_id, base_id, &n_zones);
    EXPECT_EQ(my_base.CountZones(), n_zones);
    for (int zone_id = 1; zone_id <= n_zones; ++zone_id) {
      auto& my_zone = my_base.GetZone(zone_id);
      auto& cg_zone = zone_info[index++];
      EXPECT_STREQ(my_zone.GetName().c_str(), cg_zone.name.c_str());
      EXPECT_EQ(my_zone.GetId(), cg_zone.id);
      EXPECT_EQ(my_zone.GetVertexSize(), cg_zone.x.size());
      EXPECT_EQ(my_zone.GetCellSize(), cg_zone.cell_size);
      // read coordinates
      auto n_nodes = my_zone.GetVertexSize();
      auto& my_coor = my_zone.GetCoordinates();
      for (int node_id = 0; node_id < n_nodes; ++node_id) {
        EXPECT_DOUBLE_EQ(my_coor.x[node_id], cg_zone.x[node_id]);
        EXPECT_DOUBLE_EQ(my_coor.y[node_id], cg_zone.y[node_id]);
        EXPECT_DOUBLE_EQ(my_coor.z[node_id], cg_zone.z[node_id]);
      }
    }
  }
  cg_close(file_id);
}

// TEST_F(ReaderTest, ReadFromFile) {
//   int file_id;
//   auto file_name = test_data_dir_ + "ugrid_2d.cgns";
//   if (cg_open(file_name.c_str(), CG_MODE_READ, &file_id)) {
//     cg_error_exit();
//   }
//   else {
//     int n_bases, n_zones;
//     cg_nbases(file_id, &n_bases);
//     if (n_bases) {
//       cg_nzones(file_id, n_bases, &n_zones);
//       std::printf("There are %d `Base`s and %d `Zone`s.\n", n_bases, n_zones);
//     }
//     cg_close(file_id);
//   }
// }

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
