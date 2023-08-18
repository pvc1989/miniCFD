// Copyright 2021 PEI Weicheng and YANG Minghao and JIANG Yuyan

#include <cstdio>
#include <string>
#include <vector>

#include "cgnslib.h"
#include "gtest/gtest.h"

#include "mini/mesh/cgns.hpp"
#include "mini/input/path.hpp"  // defines TEST_INPUT_DIR

namespace mini {
namespace mesh {
namespace cgns {

class TestTypes : public ::testing::Test {
 protected:
  // common types
  using myFile = File<double>;
  using cgField = std::vector<double>;
  struct cgSolution {
    std::string name; int id; GridLocation location;
    std::map<std::string, cgField> fields;
    cgSolution(char* sn, int si, GridLocation lc)
        : name(sn), id(si), location(lc) {
    }
  };
  struct cgSection {
    std::string name; int id, first, last, n_boundary;
    ElementType type;
    std::vector<int> i_node_list;
    cgSection(char* sn, int si, int fi, int la, int nb, ElementType ty)
        : name(sn), id(si), first(fi), last(la), n_boundary(nb), type(ty),
          i_node_list((last-first+1) * cgns::CountNodesByType(ty)) {
    }
  };
  struct cgZone {
    std::string name; int id, vertex_size, cell_size;
    std::vector<double> x, y, z;
    std::map<int, cgSection> sections;
    std::vector<cgSolution> solutions;
    cgZone(char* zn, int zi, cgsize_t* zone_size)
        : name(zn), id(zi), vertex_size(zone_size[0]), cell_size(zone_size[1]),
          x(zone_size[0]), y(zone_size[0]), z(zone_size[0]) {
    }
  };
  // common data
  std::string const abs_path_{std::string(TEST_INPUT_DIR) + "/ugrid_2d.cgns"};
  double const eps_ = 0.00001;
};
TEST_F(TestTypes, Constructors) {
  auto files = std::vector<myFile>();
  // the absolute path version
  files.emplace_back(abs_path_);
  // the dir (with or without '/') + name version
  files.emplace_back(std::string(TEST_INPUT_DIR) + "/", "ugrid_2d.cgns");
  files.emplace_back(TEST_INPUT_DIR, "ugrid_2d.cgns");
}
TEST_F(TestTypes, ReadBase) {
  // read by mini::mesh::cgns
  auto file = myFile(abs_path_);
  file.ReadBases();
  // read by cgnslib
  int i_file{-1};
  cg_open(abs_path_.c_str(), CG_MODE_READ, &i_file);
  int n_bases{-1};
  cg_nbases(i_file, &n_bases);
  struct BaseInfo {
    std::string name; int id, cell_dim, phys_dim;
    BaseInfo(char* bn, int bi, int cd, int pd)
        : name(bn), id(bi), cell_dim(cd), phys_dim(pd) {}
  };
  auto base_info = std::vector<BaseInfo>();
  for (int base_id = 1; base_id <= n_bases; ++base_id) {
    char base_name[33];
    int cell_dim{-1}, phys_dim{-1};
    cg_base_read(i_file, base_id, base_name, &cell_dim, &phys_dim);
    base_info.emplace_back(base_name, base_id, cell_dim, phys_dim);
  }
  cg_close(i_file);
  // compare result
  EXPECT_EQ(file.CountBases(), n_bases);
  for (auto &base : base_info) {
    auto &my_base = file.GetBase(base.id);
    EXPECT_STREQ(my_base.name().c_str(), base.name.c_str());
    EXPECT_EQ(my_base.GetCellDim(), base.cell_dim);
    EXPECT_EQ(my_base.GetPhysDim(), base.phys_dim);
  }
}
TEST_F(TestTypes, ReadCoordinates) {
  // read by mini::mesh::cgns
  auto file = myFile(abs_path_);
  file.ReadBases();
  {
    // check coordinates in zones[1]
    auto &coordinates = file.GetBase(1).GetZone(1).GetCoordinates();
    auto &x = coordinates.x();
    EXPECT_DOUBLE_EQ(x.at(0), -2.0);
    EXPECT_DOUBLE_EQ(x.at(1), -2.0);
    EXPECT_NEAR(x.at(371), -1.926790, eps_);
    EXPECT_NEAR(x.at(372), -1.926790, eps_);
    auto &y = coordinates.y();
    EXPECT_DOUBLE_EQ(y.at(0), +1.0);
    EXPECT_DOUBLE_EQ(y.at(1), -1.0);
    EXPECT_NEAR(y.at(371), -0.926795, eps_);
    EXPECT_NEAR(y.at(372), +0.926795, eps_);
    auto &z = coordinates.z();
    EXPECT_DOUBLE_EQ(z.at(0), 0.0);
    EXPECT_DOUBLE_EQ(z.at(1), 0.0);
    EXPECT_DOUBLE_EQ(z.at(371), 0.0);
    EXPECT_DOUBLE_EQ(z.at(372), 0.0);
  }
  {
    // check coordinates in zones[2]
    auto &coordinates = file.GetBase(1).GetZone(2).GetCoordinates();
    auto &x = coordinates.x();
    EXPECT_DOUBLE_EQ(x.at(0), 2.0);
    EXPECT_DOUBLE_EQ(x.at(1), 0.0);
    EXPECT_NEAR(x.at(582), -0.072475, eps_);
    EXPECT_NEAR(x.at(583), -0.163400, eps_);
    auto &y = coordinates.y();
    EXPECT_DOUBLE_EQ(y.at(0), -1.0);
    EXPECT_DOUBLE_EQ(y.at(1), +1.0);
    EXPECT_NEAR(y.at(582), +0.927525, eps_);
    EXPECT_NEAR(y.at(583), -0.375511, eps_);
    auto &z = coordinates.z();
    EXPECT_DOUBLE_EQ(z.at(0), 0.0);
    EXPECT_DOUBLE_EQ(z.at(1), 0.0);
    EXPECT_DOUBLE_EQ(z.at(582), 0.0);
    EXPECT_DOUBLE_EQ(z.at(583), 0.0);
  }
}
TEST_F(TestTypes, ReadSections) {
  // read by mini::mesh::cgns
  auto file = myFile(abs_path_);
  file.ReadBases();
  {
    auto &section = file.GetBase(1).GetZone(1).GetSection(1);
    EXPECT_EQ(section.CellIdMin(), 1);
    EXPECT_EQ(section.CellIdMax(), 673);
    EXPECT_EQ(section.CountCells(), 673);
    EXPECT_EQ(section.name(), "3_S_5_10");
    EXPECT_EQ(section.type(), CGNS_ENUMV(TRI_3));
    const cgsize_t* array;  // head of 0-based-node-id list
    array = section.GetNodeIdListByOneBasedCellId(1);
    EXPECT_EQ(array, section.GetNodeIdListByNilBasedRow(0));
    EXPECT_EQ(array[0], 43);
    EXPECT_EQ(array[1], 155);
    EXPECT_EQ(array[2], 154);
    array = section.GetNodeIdListByOneBasedCellId(673);
    auto row = section.CountCells() - 1;
    EXPECT_EQ(array, section.GetNodeIdListByNilBasedRow(row));
    EXPECT_EQ(array[0], 102);
    EXPECT_EQ(array[1], 196);
    EXPECT_EQ(array[2], 98);
  }
  {
    auto &section = file.GetBase(1).GetZone(2).GetSection(1);
    EXPECT_EQ(section.CellIdMin(), 1);
    EXPECT_EQ(section.CellIdMax(), 271);
    EXPECT_EQ(section.CountCells(), 271);
    EXPECT_EQ(section.name(), "3_S_5_11");
    EXPECT_EQ(section.type(), CGNS_ENUMV(TRI_3));
    const cgsize_t* array;  // head of 0-based-node-id list
    array = section.GetNodeIdListByOneBasedCellId(1);
    EXPECT_EQ(array, section.GetNodeIdListByNilBasedRow(0));
    EXPECT_EQ(array[0], 347);
    EXPECT_EQ(array[1], 510);
    EXPECT_EQ(array[2], 349);
    array = section.GetNodeIdListByOneBasedCellId(271);
    auto row = section.CountCells() - 1;
    EXPECT_EQ(array, section.GetNodeIdListByNilBasedRow(row));
    EXPECT_EQ(array[0], 367);
    EXPECT_EQ(array[1], 503);
    EXPECT_EQ(array[2], 492);
  }
  {
    auto &section = file.GetBase(1).GetZone(2).GetSection(2);
    EXPECT_EQ(section.CellIdMin(), 272);
    EXPECT_EQ(section.CellIdMax(), 671);
    EXPECT_EQ(section.CountCells(), 400);
    EXPECT_EQ(section.name(), "4_S_9_12");
    EXPECT_EQ(section.type(), CGNS_ENUMV(QUAD_4));
    const cgsize_t* array;  // head of 0-based-node-id list
    array = section.GetNodeIdListByOneBasedCellId(272);
    EXPECT_EQ(array, section.GetNodeIdListByNilBasedRow(0));
    EXPECT_EQ(array[0], 4);
    EXPECT_EQ(array[1], 456);
    EXPECT_EQ(array[2], 416);
    EXPECT_EQ(array[3], 543);
    array = section.GetNodeIdListByOneBasedCellId(671);
    auto row = section.CountCells() - 1;
    EXPECT_EQ(array, section.GetNodeIdListByNilBasedRow(row));
    EXPECT_EQ(array[0], 467);
    EXPECT_EQ(array[1], 2);
    EXPECT_EQ(array[2], 469);
    EXPECT_EQ(array[3], 142);
  }
}
TEST_F(TestTypes, MergeAndSplitSections) {
  auto file = myFile(abs_path_);
  file.ReadBases();
  auto &zone = file.GetBase(1).GetZone(2);
  auto n_sect = zone.CountSections();
  // Merge the two 3D sections:
  zone.MergeSections({1, 2});
  EXPECT_EQ(n_sect, 1 + zone.CountSections());
  auto &section = zone.GetSection(1);
  EXPECT_EQ(section.type(), CGNS_ENUMV(MIXED));
  EXPECT_EQ(section.CellIdMin(), 1);
  EXPECT_EQ(section.CellIdMax(), 671);
  const cgsize_t* array;  // head of 1-based-node-id list
  array = section.GetNodeIdListByOneBasedCellId(1);
  EXPECT_EQ(array[0], CGNS_ENUMV(TRI_3));
  EXPECT_EQ(array[1], 347);
  EXPECT_EQ(array[2], 510);
  EXPECT_EQ(array[3], 349);
  array = section.GetNodeIdListByOneBasedCellId(271);
  EXPECT_EQ(array[0], CGNS_ENUMV(TRI_3));
  EXPECT_EQ(array[1], 367);
  EXPECT_EQ(array[2], 503);
  EXPECT_EQ(array[3], 492);
  array = section.GetNodeIdListByOneBasedCellId(272);
  EXPECT_EQ(array[0], CGNS_ENUMV(QUAD_4));
  EXPECT_EQ(array[1], 4);
  EXPECT_EQ(array[2], 456);
  EXPECT_EQ(array[3], 416);
  EXPECT_EQ(array[4], 543);
  array = section.GetNodeIdListByOneBasedCellId(671);
  EXPECT_EQ(array[0], CGNS_ENUMV(QUAD_4));
  EXPECT_EQ(array[1], 467);
  EXPECT_EQ(array[2], 2);
  EXPECT_EQ(array[3], 469);
  EXPECT_EQ(array[4], 142);
  // Split the section just merged:
  zone.SplitSections({1});
  EXPECT_EQ(n_sect, zone.CountSections());
  {
    auto &section = zone.GetSection(1);
    EXPECT_EQ(section.type(), CGNS_ENUMV(TRI_3));
    EXPECT_EQ(section.CellIdMin(), 1);
    EXPECT_EQ(section.CellIdMax(), 271);
    EXPECT_EQ(section.CountCells(), 271);
    const cgsize_t* array;  // head of 0-based-node-id list
    array = section.GetNodeIdListByOneBasedCellId(1);
    EXPECT_EQ(array[0], 347);
    EXPECT_EQ(array[1], 510);
    EXPECT_EQ(array[2], 349);
    array = section.GetNodeIdListByOneBasedCellId(271);
    EXPECT_EQ(array[0], 367);
    EXPECT_EQ(array[1], 503);
    EXPECT_EQ(array[2], 492);
  }
  {
    auto &section = zone.GetSection(2);
    EXPECT_EQ(section.type(), CGNS_ENUMV(QUAD_4));
    EXPECT_EQ(section.CellIdMin(), 272);
    EXPECT_EQ(section.CellIdMax(), 671);
    EXPECT_EQ(section.CountCells(), 400);
    array = section.GetNodeIdListByOneBasedCellId(272);
    EXPECT_EQ(array[0], 4);
    EXPECT_EQ(array[1], 456);
    EXPECT_EQ(array[2], 416);
    EXPECT_EQ(array[3], 543);
    array = section.GetNodeIdListByOneBasedCellId(671);
    EXPECT_EQ(array[0], 467);
    EXPECT_EQ(array[1], 2);
    EXPECT_EQ(array[2], 469);
    EXPECT_EQ(array[3], 142);
  }
}
TEST_F(TestTypes, ReadZone) {
  // read by mini::mesh::cgns
  auto file = myFile(abs_path_);
  file.ReadBases();
  // read by cgnslib
  int i_file{-1};
  cg_open(abs_path_.c_str(), CG_MODE_READ, &i_file);
  auto zone_info = std::vector<cgZone>();
  int n_bases{-1};
  cg_nbases(i_file, &n_bases);
  for (int base_id = 1; base_id <= n_bases; ++base_id) {
    int n_zones{-1};
    cg_nzones(i_file, base_id, &n_zones);
    for (int i_zone = 1; i_zone <= n_zones; ++i_zone) {
      char zone_name[33];
      cgsize_t zone_size[3][1];
      cg_zone_read(i_file, base_id, i_zone, zone_name, zone_size[0]);
      zone_info.emplace_back(zone_name, i_zone, zone_size[0]);
    }
  }
  cg_close(i_file);
  // compare result
  cg_open(abs_path_.c_str(), CG_MODE_READ, &i_file);
  EXPECT_EQ(file.CountBases(), n_bases);
  int index = 0;
  for (int base_id = 1; base_id <= n_bases; ++base_id) {
    auto &my_base = file.GetBase(base_id);
    int n_zones{0};
    cg_nzones(i_file, base_id, &n_zones);
    EXPECT_EQ(my_base.CountZones(), n_zones);
    for (int i_zone = 1; i_zone <= n_zones; ++i_zone) {
      auto &my_zone = my_base.GetZone(i_zone);
      auto &cg_zone = zone_info[index++];
      EXPECT_STREQ(my_zone.name().c_str(), cg_zone.name.c_str());
      EXPECT_EQ(my_zone.id(), cg_zone.id);
      EXPECT_EQ(my_zone.CountNodes(), cg_zone.x.size());
      EXPECT_EQ(my_zone.CountCells(), cg_zone.cell_size);
    }
  }
  cg_close(i_file);
}
TEST_F(TestTypes, ReadSolution) {
  // read by mini::mesh::cgns
  auto file = myFile(abs_path_);
  file.ReadBases();
  // read by cgnslib
  int i_file{-1};
  cg_open(abs_path_.c_str(), CG_MODE_READ, &i_file);
  int base_id{1}, i_zone{1}, n_sols{0};
  char zone_name[33];
  cgsize_t zone_size[3][1];
  cg_zone_read(i_file, base_id, i_zone, zone_name, zone_size[0]);
  auto cg_zone = cgZone(zone_name, i_zone, zone_size[0]);
  cg_nsols(i_file, base_id, i_zone, &n_sols);
  cg_zone.solutions.reserve(n_sols);
  for (int sol_id = 1; sol_id <= n_sols; ++sol_id) {
    char sol_name[33];
    GridLocation location;
    cg_sol_info(i_file, base_id, i_zone, sol_id, sol_name, &location);
    auto &cg_solution = cg_zone.solutions.emplace_back(
        sol_name, sol_id, location);
    // read field
    int n_fields;
    cg_nfields(i_file, base_id, i_zone, sol_id, &n_fields);
    for (int field_id = 1; field_id <= n_fields; ++field_id) {
      DataType datatype;
      char field_name[33];
      cg_field_info(i_file, base_id, i_zone, sol_id, field_id, &datatype,
                    field_name);
      cgsize_t first{1}, last{1};
      if (location == CGNS_ENUMV(Vertex)) {
        last = cg_zone.vertex_size;
      } else if (location == CGNS_ENUMV(CellCenter)) {
        last = cg_zone.cell_size;
      }
      std::string name = std::string(field_name);
      cg_solution.fields.emplace(name, cgField(last));
      cg_field_read(i_file, base_id, i_zone, sol_id, field_name,
                    datatype, &first, &last, cg_solution.fields[name].data());
    }
  }
  cg_close(i_file);
  // compara flow solutions
  auto &my_zone = file.GetBase(base_id).GetZone(i_zone);
  for (int sol_id = 1; sol_id <= n_sols; ++sol_id) {
    auto &my_sol = my_zone.GetSolution(sol_id);
    auto &cg_sol = cg_zone.solutions.at(sol_id-1);
    EXPECT_EQ(my_sol.id(), cg_sol.id);
    EXPECT_STREQ(my_sol.name().c_str(), cg_sol.name.c_str());
    EXPECT_EQ(my_sol.CountFields(), cg_sol.fields.size());
    for (int i = 0; i < my_sol.CountFields(); ++i) {
      auto &field = my_sol.GetField(i);
      auto name = field.name();
      for (int index = 0; index < field.size(); ++index) {
        EXPECT_DOUBLE_EQ(field.at(index), cg_sol.fields[name].at(index));
      }
    }
  }
}

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
