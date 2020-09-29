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
  double abs_error = 0.00001;
  using Field = std::vector<double>;
  struct Solution {
    std::string name; int id; CGNS_ENUMT(GridLocation_t) location;
    std::map<std::string, Field> fields;
    Solution(char* sn, int si, CGNS_ENUMT(GridLocation_t) lc)
        : name(sn), id(si), location(lc) {}
  };
  struct Section {
    std::string name; int id, first, last, n_boundary;
    CGNS_ENUMT(ElementType_t) type;
    std::vector<int> connectivity;
    Section(char* sn, int si, int fi, int la, int nb, CGNS_ENUMT(ElementType_t) ty)
        : name(sn), id(si), first(fi), last(la), n_boundary(nb), type(ty),
          connectivity((last-first+1) * CountNodesByType(ty)) {}
  };
  struct ZoneInfo {
    std::string name; int id, vertex_size, cell_size;
    std::vector<double> x, y, z;
    std::map<int, Section> sections;
    std::vector<Solution> solutions;
    ZoneInfo(char* zn, int zi, int* zone_size) 
        : name(zn), id(zi), vertex_size(zone_size[0]), cell_size(zone_size[1]),
          x(zone_size[0]), y(zone_size[0]), z(zone_size[0]) {}
  };
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
TEST_F(ReaderTest, ReadCoordinates) {
  auto file_name = test_data_dir_ + "ugrid_2d.cgns";
  // read by mini::mesh::cgns
  reader.ReadFromFile(file_name);
  auto mesh = reader.GetMesh();
  {
    // check coordinates in zones[1]
    auto& coordinates = mesh->GetBase(1).GetZone(1).GetCoordinates();
    auto& x = coordinates.x;
    EXPECT_DOUBLE_EQ(x.at(0), -2.0);
    EXPECT_DOUBLE_EQ(x.at(1), -2.0);
    EXPECT_NEAR(x.at(371), -1.926790, abs_error);
    EXPECT_NEAR(x.at(372), -1.926790, abs_error);
    auto& y = coordinates.y;
    EXPECT_DOUBLE_EQ(y.at(0), +1.0);
    EXPECT_DOUBLE_EQ(y.at(1), -1.0);
    EXPECT_NEAR(y.at(371), -0.926795, abs_error);
    EXPECT_NEAR(y.at(372), +0.926795, abs_error);
    auto& z = coordinates.z;
    EXPECT_DOUBLE_EQ(z.at(0), 0.0);
    EXPECT_DOUBLE_EQ(z.at(1), 0.0);
    EXPECT_DOUBLE_EQ(z.at(371), 0.0);
    EXPECT_DOUBLE_EQ(z.at(372), 0.0);
  }
  {
    // check coordinates in zones[2]
    auto& coordinates = mesh->GetBase(1).GetZone(2).GetCoordinates();
    auto& x = coordinates.x;
    EXPECT_DOUBLE_EQ(x.at(0), 2.0);
    EXPECT_DOUBLE_EQ(x.at(1), 0.0);
    EXPECT_NEAR(x.at(582), -0.072475, abs_error);
    EXPECT_NEAR(x.at(583), -0.163400, abs_error);
    auto& y = coordinates.y;
    EXPECT_DOUBLE_EQ(y.at(0), -1.0);
    EXPECT_DOUBLE_EQ(y.at(1), +1.0);
    EXPECT_NEAR(y.at(582), +0.927525, abs_error);
    EXPECT_NEAR(y.at(583), -0.375511, abs_error);
    auto& z = coordinates.z;
    EXPECT_DOUBLE_EQ(z.at(0), 0.0);
    EXPECT_DOUBLE_EQ(z.at(1), 0.0);
    EXPECT_DOUBLE_EQ(z.at(582), 0.0);
    EXPECT_DOUBLE_EQ(z.at(583), 0.0);
  }
}
TEST_F(ReaderTest, ReadZone) {
  auto file_name = test_data_dir_ + "ugrid_2d.cgns";
  // read by cgnslib
  int file_id{-1};
  cg_open(file_name.c_str(), CG_MODE_READ, &file_id);
  auto zone_info = std::vector<ZoneInfo>();
  int n_bases{-1};
  cg_nbases(file_id, &n_bases);
  for (int base_id = 1; base_id <= n_bases; ++base_id) {
    int n_zones{-1};
    cg_nzones(file_id, base_id, &n_zones);
    for (int zone_id = 1; zone_id <= n_zones; ++zone_id) {
      char zone_name[33];
      cgsize_t zone_size[3][1];
      cg_zone_read(file_id, base_id, zone_id, zone_name, zone_size[0]);
      auto& cg_zone = zone_info.emplace_back(zone_name, zone_id, zone_size[0]);
      // read coordinates
      cgsize_t first = 1;
      cgsize_t last = cg_zone.x.size();
      cg_coord_read(file_id, base_id, zone_id, "CoordinateX",
                    CGNS_ENUMV(RealDouble), &first, &last, cg_zone.x.data());
      cg_coord_read(file_id, base_id, zone_id, "CoordinateY",
                    CGNS_ENUMV(RealDouble), &first, &last, cg_zone.y.data());
      cg_coord_read(file_id, base_id, zone_id, "CoordinateZ",
                    CGNS_ENUMV(RealDouble), &first, &last, cg_zone.z.data());
      // read elements
      int n_sections;
      cg_nsections(file_id, base_id, zone_id, &n_sections);
      for (int section_id = 1; section_id <= n_sections; ++section_id) {
        char section_name[33];
        CGNS_ENUMT(ElementType_t) element_type;
        int first, last, n_boundary, parent_flag;
        cg_section_read(file_id, base_id, zone_id, section_id, section_name,
                        &element_type, &first, &last, &n_boundary, &parent_flag);
        Section cg_section(section_name, section_id, first, last, n_boundary,
                           element_type);
        int parent_data;
        cg_elements_read(file_id, base_id, zone_id, section_id,
                         cg_section.connectivity.data(), &parent_data);
        cg_zone.sections.insert({section_id, cg_section});
      }
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
      EXPECT_EQ(my_zone.CountNodes(), cg_zone.x.size());
      EXPECT_EQ(my_zone.CountCells(), cg_zone.cell_size);
      // compare elements
      auto n_sections = my_zone.CountSections();
      for (int section_id = 1; section_id <= n_sections; ++section_id) {
        auto& my_section = my_zone.GetSection(section_id);
        auto& cg_section = cg_zone.sections.at(section_id);
        EXPECT_EQ(my_section.id, cg_section.id);
        EXPECT_EQ(my_section.first, cg_section.first);
        EXPECT_EQ(my_section.last, cg_section.last);
        EXPECT_STREQ(my_section.name.c_str(), cg_section.name.c_str());
        EXPECT_EQ(my_section.connectivity.size(),
                  cg_section.connectivity.size());
        int n_nodes = my_section.connectivity.size();
        for (int index = 0; index < n_nodes; ++index) {
          EXPECT_EQ(my_section.connectivity.at(index),
                    cg_section.connectivity.at(index));
        }
      }
    }
  }
  cg_close(file_id);
}

TEST_F(ReaderTest, ReadSolution) {
  auto file_name = test_data_dir_ + "fixed_grid.cgns";
  // read by mini::mesh::cgns
  reader.ReadFromFile(file_name);
  auto mesh = reader.GetMesh();
  // read by cgnslib
  int file_id{-1};
  cg_open(file_name.c_str(), CG_MODE_READ, &file_id);
  int base_id{1}, zone_id{1}, n_sols{0};
  char zone_name[33];
  cgsize_t zone_size[3][1];
  cg_zone_read(file_id, base_id, zone_id, zone_name, zone_size[0]);
  auto cg_zone = ZoneInfo(zone_name, zone_id, zone_size[0]);
  cg_nsols(file_id, base_id, zone_id, &n_sols);
  cg_zone.solutions.reserve(n_sols);
  for (int sol_id = 1; sol_id <= n_sols; ++sol_id) {
    char sol_name[33];
    CGNS_ENUMT(GridLocation_t) location;
    cg_sol_info(file_id, base_id, zone_id, sol_id, sol_name, &location);
    auto& cg_solution = cg_zone.solutions.emplace_back(sol_name, sol_id, location);
    // read field
    int n_fields;
    cg_nfields(file_id, base_id, zone_id, sol_id, &n_fields);
    for (int field_id = 1; field_id <= n_fields; ++field_id) {
      CGNS_ENUMT(DataType_t) datatype;
      char field_name[33];
      cg_field_info(file_id, base_id, zone_id, sol_id, field_id, &datatype,
                    field_name);
      int first{1}, last{1};
      if (location == CGNS_ENUMV(Vertex)) {
        last = cg_zone.vertex_size;
      } else if (location == CGNS_ENUMV(CellCenter)) {
        last = cg_zone.cell_size;
      }
      std::string name = std::string(field_name);
      cg_solution.fields.emplace(name, Field(last));
      cg_field_read(file_id, base_id, zone_id, sol_id, field_name,
                    datatype, &first, &last, cg_solution.fields[name].data());
    }
  }
  cg_close(file_id);
  // compara flow solutions
  auto& my_zone = mesh->GetBase(base_id).GetZone(zone_id);
  for (int sol_id = 1; sol_id <= n_sols; ++sol_id) {
    auto& my_sol = my_zone.GetSolution(sol_id);
    auto& cg_sol = cg_zone.solutions.at(sol_id-1);
    EXPECT_EQ(my_sol.id, cg_sol.id);
    EXPECT_STREQ(my_sol.name.c_str(), cg_sol.name.c_str());
    EXPECT_EQ(my_sol.fields.size(), cg_sol.fields.size());
    for (auto& [name, field] : my_sol.fields) {
      EXPECT_EQ(field.size(), cg_sol.fields[name].size());
      for (int index = 0; index < field.size(); ++index) {
        EXPECT_DOUBLE_EQ(field.at(index), cg_sol.fields[name].at(index));
      }
    }
  }
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
