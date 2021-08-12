// Copyright 2019 Weicheng Pei and Minghao Yang

#include <cassert>
#include <cstdio>
#include <map>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "cgnslib.h"
#include "metis.h"
#include "gtest/gtest.h"

#include "mini/mesh/filter/cgns_to_metis.hpp"
#include "mini/mesh/cgns/shuffler.hpp"
#include "mini/mesh/cgns/format.hpp"
#include "mini/mesh/metis/format.hpp"
#include "mini/mesh/metis/partitioner.hpp"
#include "mini/data/path.hpp"  // defines PROJECT_BINARY_DIR

namespace mini {
namespace mesh {

class ShufflerTest : public ::testing::Test {
 protected:
  using CgnsFile = mini::mesh::cgns::File<double>;
  using MetisMesh = metis::Mesh<idx_t>;
  using CSRM = mini::mesh::metis::SparseMatrix<idx_t>;
  using FilterType = mini::mesh::filter::CgnsToMetis<double, idx_t>;
  using FieldType = mini::mesh::cgns::Field<double>;
  std::string const test_data_dir_{TEST_DATA_DIR};
  std::string const current_binary_dir_{
      std::string(PROJECT_BINARY_DIR) + std::string("/test/mesh")};
  static void SetCellPartData(
      const FilterType& filter, const std::vector<idx_t>& cell_parts,
      CgnsFile* cgns_mesh);
};
void ShufflerTest::SetCellPartData(
    const FilterType& filter, const std::vector<idx_t>& cell_parts,
    CgnsFile* cgns_mesh) {
  auto& zone_to_sections = filter.cgns_to_metis_for_cells;
  auto& base = cgns_mesh->GetBase(1); int n_zones = base.CountZones();
  for (int zone_id = 1; zone_id <= n_zones; ++zone_id) {
    auto& zone = base.GetZone(zone_id);
    auto& section_to_cells = zone_to_sections.at(zone_id);
    auto& sol = zone.AddSolution("CellData", CGNS_ENUMV(CellCenter));
    auto& field = sol.AddField("CellPart");
    int n_sections = zone.CountSections();
    for (int section_id = 1; section_id <= n_sections; ++section_id) {
      auto& section = zone.GetSection(section_id);
      if (section.dim() != base.GetCellDim())
        continue;
      auto& cells_local_to_global = section_to_cells.at(section_id);
      int n_cells = section.CountCells();
      int range_min{section.CellIdMin()};
      int range_max{section.CellIdMax()};
      EXPECT_EQ(n_cells - 1, range_max - range_min);
      for (int cgns_cell_id = range_min; cgns_cell_id <= range_max;) {
        auto metis_cell_id = cells_local_to_global.at(cgns_cell_id - range_min);
        field.at(cgns_cell_id++) = cell_parts[metis_cell_id];
      }
    }
  }
}
TEST_F(ShufflerTest, ShuffleByParts) {
  // Reorder the indices by parts
  int n = 10;
  std::vector<int> old_index{2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  std::vector<int>     parts{1, 0, 0, 2, 0, 1, 2, 1,  0,  2};
  std::vector<int> new_order(n);
  ReorderByParts<int>(parts, new_order.data());
  std::vector<int> expected_new_order{1, 2, 4, 8, 0, 5, 7, 3, 6, 9};
  for (int i = 0; i < n; ++i) {
    EXPECT_EQ(new_order[i], expected_new_order[i]);
  }
  // Shuffle data array by new order
  std::vector<double> old_array{1.0, 0.0, 0.1, 2.0, 0.2,
                                1.1, 2.1, 1.2, 0.3, 2.2};
  ShuffleDataArray<double>(new_order, old_array.data());
  std::vector<double> expected_new_array{0.0, 0.1, 0.2, 0.3, 1.0,
                                         1.1, 1.2, 2.0, 2.1, 2.2};
  for (int i = 0; i < n; ++i) {
    EXPECT_DOUBLE_EQ(old_array[i], expected_new_array[i]);
  }
  // Shuffle cell node_id_list by new order
  n = 4;
  std::vector<int> new_cell_order{1, 2, 3, 0};
  std::vector<int> node_id_list = {1, 2, 6,   2, 3, 6,   6, 3, 5,   3, 4, 5};
  int npe = 3;
  ShuffleConnectivity<int>(new_cell_order, npe, node_id_list.data());
  std::vector<int> expected_new_node_id_list{2, 3, 6,   6, 3, 5,   3, 4, 5,
                                             1, 2, 6};
  for (int i = 0; i < node_id_list.size(); ++i) {
    EXPECT_EQ(node_id_list[i], expected_new_node_id_list[i]);
  }
}
TEST_F(ShufflerTest, PartitionCgnsFile) {
  FilterType filter;
  using MeshDataType = double;
  using MetisId = idx_t;
  Shuffler<MetisId, MeshDataType> shuffler;
  auto old_file_name = test_data_dir_ + "/ugrid_2d.cgns";
  auto new_file_name = current_binary_dir_ + "/shuffled_ugrid_2d.cgns";
  auto cgns_mesh = CgnsFile(old_file_name);
  cgns_mesh.ReadBases();
  auto metis_mesh = filter.Filter(cgns_mesh);
  int n_parts{8}, n_common_nodes{2}, edge_cut{0};
  std::vector<idx_t> null_vector_of_idx;
  std::vector<float> null_vector_of_real;
  std::vector<idx_t> cell_parts, node_parts;
  PartMesh(metis_mesh,
      null_vector_of_idx/* computational cost */,
      null_vector_of_idx/* communication size */,
      n_common_nodes, n_parts,
      null_vector_of_real/* weight of each part */,
      null_vector_of_idx/* options */,
      &edge_cut, &cell_parts, &node_parts);
  shuffler.SetNumParts(n_parts);
  shuffler.SetCellParts(&cell_parts);
  shuffler.SetMetisMesh(&metis_mesh);
  shuffler.SetFilter(&filter);
  SetCellPartData(filter, cell_parts, &cgns_mesh);
  shuffler.ShuffleMesh(&cgns_mesh);
  cgns_mesh.Write(new_file_name, 2);
  //
  auto& base = cgns_mesh.GetBase(1);
  int n_zones = base.CountZones();
  // part_to_nodes[part_id][zone_id] == pair<begin_id, end_id>;
  // part_to_cells[part_id][zone_id][section_id] == pair<begin_id, end_id>;
  auto part_to_nodes = std::vector<std::vector<std::pair<int, int>>>(n_parts);
  auto part_to_cells = std::vector<std::vector<std::vector<std::pair<int, int>>>>(n_parts);
  for (int p = 0; p < n_parts; ++p) {
    part_to_nodes[p].resize(n_zones+1);
    part_to_cells[p].resize(n_zones+1);
    for (int z = 1; z <= n_zones; ++z) {
      part_to_cells[p][z].resize(base.GetZone(z).CountSections() + 1);
    }
  }
  for (int zid = 1; zid <= n_zones; ++zid) {
    auto& zone = base.GetZone(zid);
    auto& node_parts = zone.GetSolution(1).GetField(1);
    assert(node_parts.name() == "NodePart");
    // slice node lists by part_id
    int prev_nid = 1, prev_part = node_parts.at(prev_nid);
    int n_nodes = zone.CountNodes();
    for (int curr_nid = prev_nid+1; curr_nid <= n_nodes; ++curr_nid) {
      int curr_part = node_parts.at(curr_nid);
      if (curr_part != prev_part) {
        part_to_nodes[prev_part][zid] = std::make_pair(prev_nid, curr_nid);
        prev_nid = curr_nid;
        prev_part = curr_part;
      }
    }
    part_to_nodes[prev_part][zid] = std::make_pair(prev_nid, n_nodes+1);
    // slice cell lists by part_id
    int n_cells = zone.CountCells();
    auto& sol = zone.GetSolution(2);
    assert(sol.name() == "CellData");
    auto& cell_parts = sol.GetField(1);
    for (int sid = 1; sid <= zone.CountSections(); ++sid) {
      auto& section = zone.GetSection(sid);
      int cid_min = section.CellIdMin(), cid_max = section.CellIdMax();
      int prev_cid = cid_min, prev_part = node_parts.at(prev_cid);
      for (int curr_cid = prev_cid+1; curr_cid <= cid_max; ++curr_cid) {
        int curr_part = cell_parts.at(curr_cid );
        if (curr_part != prev_part) {
          part_to_cells[prev_part][zid][sid] = std::make_pair(prev_cid, curr_cid);
          prev_cid = curr_cid;
          prev_part = curr_part;
        }
      }
      part_to_cells[prev_part][zid][sid] = std::make_pair(prev_cid, cid_max+1);
    }
  }
  // write to txts

  for (int p = 0; p < n_parts; ++p) {
    auto filename = current_binary_dir_ + "/parts/" + std::to_string(p) + ".txt";
    auto ostrm = std::ofstream(filename/* , std::ios::binary */);
    for (int z = 1; z <= n_zones; ++z) {
      auto [head, tail] = part_to_nodes[p][z];
      ostrm << z << ' ' << head << ' ' << tail << '\n';
    }
  }
}

class Partition : public ::testing::Test {
 protected:
  using MetisMesh = mini::mesh::metis::Mesh<int>;
};
TEST_F(Partition, GetNodePartsByConnectivity) {
  std::vector<int> cell_parts{2, 0, 0, 1};
  int n_nodes = 10;
  auto mesh = MetisMesh(
      {0, 4, 8, 11, 14},
      {0, 2, 3, 1,   2, 4, 5, 3,   6, 8, 7,   8, 9, 7},
      n_nodes);
  int n_parts = 3;
  auto node_parts = GetNodePartsByConnectivity<int>(
      mesh, cell_parts, n_parts, n_nodes);
  std::vector<int> expected_node_parts{2, 2, 0, 0, 0, 0, 0, 0, 0, 1};
  for (int i = 0; i < n_nodes; ++i) {
    EXPECT_EQ(node_parts[i], expected_node_parts[i]);
  }
}

}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
