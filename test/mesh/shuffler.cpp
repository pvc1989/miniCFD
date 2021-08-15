// Copyright 2019 Weicheng Pei and Minghao Yang

#include <cassert>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "mini/mesh/mapper/cgns_to_metis.hpp"
#include "mini/mesh/cgns/shuffler.hpp"
#include "mini/mesh/cgns/format.hpp"
#include "mini/mesh/metis/format.hpp"
#include "mini/mesh/metis/partitioner.hpp"
#include "mini/data/path.hpp"  // defines PROJECT_BINARY_DIR

namespace mini {
namespace mesh {

class ShufflerTest : public ::testing::Test {
 protected:
  using CgnsMesh = mini::mesh::cgns::File<double>;
  using MetisMesh = metis::Mesh<idx_t>;
  using MapperType = mini::mesh::mapper::CgnsToMetis<double, idx_t>;
  using FieldType = mini::mesh::cgns::Field<double>;
  std::string const test_data_dir_{TEST_DATA_DIR};
  std::string const current_binary_dir_{
      std::string(PROJECT_BINARY_DIR) + std::string("/test/mesh")};
  static void WriteParts(
      const MapperType& mapper, const std::vector<idx_t>& cell_parts,
      const std::vector<idx_t>& node_parts, CgnsMesh* cgns_mesh);
};
void ShufflerTest::WriteParts(
    const MapperType& mapper, const std::vector<idx_t>& cell_parts,
    const std::vector<idx_t>& node_parts, CgnsMesh* cgns_mesh) {
  auto& zone_to_sects = mapper.cgns_to_metis_for_cells;
  auto& zone_to_nodes = mapper.cgns_to_metis_for_nodes;
  auto& base = cgns_mesh->GetBase(1);
  int n_zones = base.CountZones();
  for (int zone_id = 1; zone_id <= n_zones; ++zone_id) {
    auto& zone = base.GetZone(zone_id);
    // write data on nodes
    auto& node_sol = zone.AddSolution("NodeData",
        CGNS_ENUMV(Vertex));
    auto& node_field = node_sol.AddField("NodePart");
    auto n_nodes = zone.CountNodes();
    for (int cgns_nid = 1; cgns_nid <= n_nodes; ++cgns_nid) {
      auto metis_nid = zone_to_nodes[zone_id][cgns_nid];
      node_field.at(cgns_nid) = node_parts[metis_nid];
    }
    // write data on cells
    auto& cell_sol =  zone.AddSolution("CellData",
        CGNS_ENUMV(CellCenter));
    auto& cell_field = cell_sol.AddField("CellPart");
    auto& sect_to_cells = zone_to_sects.at(zone_id);
    int n_sects = zone.CountSections();
    for (int sect_id = 1; sect_id <= n_sects; ++sect_id) {
      auto& sect = zone.GetSection(sect_id);
      if (sect.dim() != base.GetCellDim())
        continue;
      auto& cells_local_to_global = sect_to_cells.at(sect_id);
      int n_cells = sect.CountCells();
      int range_min{sect.CellIdMin()};
      int range_max{sect.CellIdMax()};
      EXPECT_EQ(n_cells - 1, range_max - range_min);
      for (int cgns_cell_id = range_min; cgns_cell_id <= range_max;) {
        auto metis_cell_id = cells_local_to_global.at(cgns_cell_id);
        cell_field.at(cgns_cell_id++) = cell_parts[metis_cell_id];
      }
    }
  }
}
TEST_F(ShufflerTest, GetNewOrderToShuffleData) {
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
  std::vector<double> old_array{1.0, 0.0, 0.1, 2.0, 0.2,
                                1.1, 2.1, 1.2, 0.3, 2.2};
  ShuffleData(new_to_old, old_array.data());
  std::vector<double> expected_new_array{0.0, 0.1, 0.2, 0.3, 1.0,
                                         1.1, 1.2, 2.0, 2.1, 2.2};
  for (int i = 0; i < n; ++i) {
    EXPECT_DOUBLE_EQ(old_array[i], expected_new_array[i]);
  }
}
TEST_F(ShufflerTest, ShuffleConnectivity) {
  // Shuffle cell node_id_list by new order
  std::vector<int> new_to_old_for_nodes{5, 4, 3, 2, 1, 0};
  std::vector<int> new_to_old_for_cells{1, 2, 3, 0};
  std::vector<int> node_id_list = {
      1, 2, 6,   2, 3, 6,   6, 3, 5,   3, 4, 5};
  int npe = 3;
  ShuffleConnectivity(
      new_to_old_for_nodes, new_to_old_for_cells, npe, node_id_list.data());
  std::vector<int> expected_new_node_id_list{
      5, 4, 1,   1, 4, 2,   4, 3, 2,   6, 5, 1};
  for (int i = 0; i < node_id_list.size(); ++i) {
    EXPECT_EQ(node_id_list[i], expected_new_node_id_list[i]);
  }
}
TEST_F(ShufflerTest, PartitionCgnsMesh) {
  MapperType mapper;
  using MeshDataType = double;
  using MetisId = idx_t;
  auto old_file_name = test_data_dir_ + "/ugrid_2d.cgns";
  auto new_file_name = current_binary_dir_ + "/ugrid_2d_shuffled.cgns";
  auto cgns_mesh = CgnsMesh(old_file_name);
  cgns_mesh.ReadBases();
  auto metis_mesh = mapper.Map(cgns_mesh);
  EXPECT_TRUE(mapper.IsValid());
  int n_parts{8}, n_common_nodes{2}, edge_cut{0};
  std::vector<idx_t> null_vector_of_idx;
  std::vector<real_t> null_vector_of_real;
  auto graph = metis::MeshToDual(metis_mesh, n_common_nodes, 0);
  std::vector<idx_t> cell_parts;
  metis::PartGraphKway(1, graph,
      null_vector_of_idx/* weight of each cell */,
      null_vector_of_idx/* communication size */,
      null_vector_of_idx/* weight of each edge (in dual graph) */,
      n_parts, null_vector_of_real/* weight of each part */,
      null_vector_of_real/* unbalance tolerance */,
      null_vector_of_idx/* options */, &edge_cut, &cell_parts);
  std::vector<idx_t> node_parts = GetNodePartsByConnectivity(
      metis_mesh, cell_parts, n_parts, metis_mesh.CountNodes());
  // PartMesh(metis_mesh,
  //     null_vector_of_idx/* computational cost */,
  //     null_vector_of_idx/* communication size */,
  //     n_common_nodes, n_parts,
  //     null_vector_of_real/* weight of each part */,
  //     null_vector_of_idx/* options */,
  //     &edge_cut, &cell_parts, &node_parts);
  auto shuffler = Shuffler<MetisId, MeshDataType>(n_parts, cell_parts,
      node_parts);
  WriteParts(mapper, cell_parts, node_parts, &cgns_mesh);
  shuffler.Shuffle(&cgns_mesh, &mapper);
  EXPECT_TRUE(mapper.IsValid());
  cgns_mesh.Write(new_file_name, 2);

  // Write Partition txts.
  cgns_mesh = CgnsMesh(new_file_name);
  cgns_mesh.ReadBases();
  auto& base = cgns_mesh.GetBase(1);
  int n_zones = base.CountZones();
  // auto [begin_nid, end_nid] = part_to_nodes[part_id][zone_id];
  // auto [begin_cid, end_cid] = part_to_cells[part_id][zone_id][sect_id];
  auto part_to_nodes = std::vector<std::vector<std::pair<int, int>>>(n_parts);
  auto part_to_cells = std::vector<std::vector<std::vector<
      std::pair<int, int>>>>(n_parts);
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
    assert(cell_parts.name() == "CellPart");
    for (int sid = 1; sid <= zone.CountSections(); ++sid) {
      auto& sect = zone.GetSection(sid);
      if (sect.dim() != base.GetCellDim())
        continue;
      int cid_min = sect.CellIdMin(), cid_max = sect.CellIdMax();
      int prev_cid = cid_min, prev_part = cell_parts.at(prev_cid);
      for (int curr_cid = prev_cid+1; curr_cid <= cid_max; ++curr_cid) {
        int curr_part = cell_parts.at(curr_cid);
        if (curr_part != prev_part) {
          part_to_cells[prev_part][zid][sid]
              = std::make_pair(prev_cid, curr_cid);
          prev_cid = curr_cid;
          prev_part = curr_part;
        }
      }
      part_to_cells[prev_part][zid][sid] = std::make_pair(prev_cid, cid_max+1);
    }
  }
  // write to txts
  for (int p = 0; p < n_parts; ++p) {
    auto filename = current_binary_dir_ + "/ugrid_2d_part_";
    filename += std::to_string(p) + ".txt";
    auto ostrm = std::ofstream(filename/* , std::ios::binary */);
    for (int z = 1; z <= n_zones; ++z) {
      auto [head, tail] = part_to_nodes[p][z];
      if (head)
        ostrm << z << ' ' << head << ' ' << tail << '\n';
      auto n_sects = part_to_cells[p][z].size() - 1;
      for (int s = 1; s <= n_sects; ++s) {
        auto [head, tail] = part_to_cells[p][z][s];
        if (head)
          ostrm << z << ' ' << s << ' ' << head << ' ' << tail << '\n';
      }
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
