// Copyright 2019 Weicheng Pei and Minghao Yang

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "mini/mesh/mapper.hpp"
#include "mini/mesh/shuffler.hpp"
#include "mini/mesh/cgns.hpp"
#include "mini/mesh/metis.hpp"
#include "mini/mesh/partitioner.hpp"
#include "mini/data/path.hpp"  // defines TEST_DATA_DIR

namespace mini {
namespace mesh {

idx_t n_parts = 4;

class ShufflerTest : public ::testing::Test {
 protected:
  using CgnsMesh = mini::mesh::cgns::File<double>;
  using MetisMesh = metis::Mesh<idx_t>;
  using MapperType = mini::mesh::mapper::CgnsToMetis<double, idx_t>;
  using FieldType = mini::mesh::cgns::Field<double>;
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
TEST_F(ShufflerTest, PartitionCgnsMesh) {
  auto case_name = std::string("double_mach_hexa");
  char cmd[1024];
  std::snprintf(cmd, sizeof(cmd), "mkdir -p %s/partition",
      case_name.c_str(), case_name.c_str());
  std::system(cmd); std::cout << "[Done] " << cmd << std::endl;
  auto old_file_name = case_name + "/original.cgns";
  auto new_file_name = case_name + "/shuffled.cgns";
  /* Generate the original cgns file: */
  std::snprintf(cmd, sizeof(cmd), "gmsh %s/%s.geo -save -o %s",
      test_data_dir_.c_str(), case_name.c_str(), old_file_name.c_str());
  std::system(cmd); std::cout << "[Done] " << cmd << std::endl;
  auto cgns_mesh = CgnsMesh(old_file_name);
  cgns_mesh.ReadBases();
  /* Partition the mesh: */
  auto mapper = MapperType();
  auto metis_mesh = mapper.Map(cgns_mesh);
  EXPECT_TRUE(mapper.IsValid());
  idx_t n_common_nodes{3};
  auto graph = metis::MeshToDual(metis_mesh, n_common_nodes);
  auto cell_parts = metis::PartGraph(graph, n_parts);
  std::vector<idx_t> node_parts = metis::GetNodeParts(
      metis_mesh, cell_parts, n_parts);
  /* Shuffle nodes and cells: */
  auto shuffler = Shuffler<idx_t, double>(n_parts, cell_parts, node_parts);
  mapper.WriteParts(cell_parts, node_parts, &cgns_mesh);
  shuffler.Shuffle(&cgns_mesh, &mapper);
  EXPECT_TRUE(mapper.IsValid());
  cgns_mesh.Write(new_file_name, 2);
  /* Reload the partitioned and shuffled mesh: */
  cgns_mesh = CgnsMesh(new_file_name);
  cgns_mesh.ReadBases();
  auto& base = cgns_mesh.GetBase(1);
  int n_zones = base.CountZones();
  /* Prepare to-be-written info for each part: */
  // auto [i_node_min, i_node_max] = part_to_nodes[i_part][i_zone];
  // auto [i_cell_min, i_cell_max] = part_to_cells[i_part][i_zone][i_sect];
  auto part_to_nodes = std::vector<std::vector<std::pair<int, int>>>(n_parts);
  auto part_to_cells
      = std::vector<std::vector<std::vector<std::pair<int, int>>>>(n_parts);
  for (int p = 0; p < n_parts; ++p) {
    part_to_nodes[p].resize(n_zones+1);
    part_to_cells[p].resize(n_zones+1);
    for (int z = 1; z <= n_zones; ++z) {
      part_to_cells[p][z].resize(base.GetZone(z).CountSections() + 1);
    }
  }
  /* Get index range of each zone's nodes and cells: */
  for (int i_zone = 1; i_zone <= n_zones; ++i_zone) {
    auto& zone = base.GetZone(i_zone);
    auto& node_field = zone.GetSolution("NodeData").GetField("NodePart");
    // slice node lists by i_part
    int prev_nid = 1, prev_part = node_field.at(prev_nid);
    int n_nodes = zone.CountNodes();
    for (int curr_nid = prev_nid+1; curr_nid <= n_nodes; ++curr_nid) {
      int curr_part = node_field.at(curr_nid);
      if (curr_part != prev_part) {
        part_to_nodes[prev_part][i_zone] = std::make_pair(prev_nid, curr_nid);
        prev_nid = curr_nid;
        prev_part = curr_part;
      }
    }  // for each node
    part_to_nodes[prev_part][i_zone] = std::make_pair(prev_nid, n_nodes+1);
    // slice cell lists by i_part
    int n_cells = zone.CountCells();
    auto& cell_field = zone.GetSolution("CellData").GetField("CellPart");
    for (int i_sect = 1; i_sect <= zone.CountSections(); ++i_sect) {
      auto& sect = zone.GetSection(i_sect);
      if (sect.dim() != base.GetCellDim())
        continue;
      int cid_min = sect.CellIdMin(), cid_max = sect.CellIdMax();
      int prev_cid = cid_min, prev_part = cell_field.at(prev_cid);
      for (int curr_cid = prev_cid+1; curr_cid <= cid_max; ++curr_cid) {
        int curr_part = cell_field.at(curr_cid);
        if (curr_part != prev_part) {
          part_to_cells[prev_part][i_zone][i_sect]
              = std::make_pair(prev_cid, curr_cid);
          prev_cid = curr_cid;
          prev_part = curr_part;
        }
      }  // for each sell
      part_to_cells[prev_part][i_zone][i_sect]
          = std::make_pair(prev_cid, cid_max+1);
    }  // for each sect
  }  // for each zone
  /* Store cell adjacency for each part: */
  // inner_adjs[i_part] = std::vector of [i_cell_small, i_cell_large]
  auto inner_adjs = std::vector<std::vector<std::pair<int, int>>>(n_parts);
  auto part_interpart_adjs
      = std::vector<std::map<int, std::vector<std::pair<int, int>>>>(n_parts);
  auto part_adj_nodes = std::vector<std::map<int, std::set<int>>>(n_parts);
  auto sendp_recvp_nodes = std::vector<std::map<int, std::set<int>>>(n_parts);
  for (int i = 0; i < metis_mesh.CountCells(); ++i) {
    auto part_i = cell_parts[i];
    int range_b = metis_mesh.range(i), range_e = metis_mesh.range(i+1);
    for (int i_range = range_b; i_range < range_e; ++i_range) {
      auto i_node = metis_mesh.nodes(i_range);
      auto node_part = node_parts[i_node];
      if (node_part != part_i) {
        part_adj_nodes[part_i][node_part].emplace(i_node);
        sendp_recvp_nodes[node_part][part_i].emplace(i_node);
      }
    }
    for (int r = graph.range(i); r < graph.range(i+1); ++r) {
      int j = graph.index(r);
      auto part_j = cell_parts[j];
      if (part_i == part_j) {
        if (i < j)
          inner_adjs[part_i].emplace_back(i, j);
      } else {
        part_interpart_adjs[part_i][part_j].emplace_back(i, j);
        int range_b = metis_mesh.range(j), range_e = metis_mesh.range(j+1);
        for (int i_range = range_b; i_range < range_e; ++i_range) {
          auto i_node = metis_mesh.nodes(i_range);
          auto node_part = node_parts[i_node];
          if (node_part != part_i) {
            part_adj_nodes[part_i][node_part].emplace(i_node);
            sendp_recvp_nodes[node_part][part_i].emplace(i_node);
          }
        }
      }
    }
  }
  /* Write part info to txts: */
  for (int p = 0; p < n_parts; ++p) {
    auto ostrm = std::ofstream(case_name + "/partition/" + std::to_string(p)
        + ".txt"/*, std::ios::binary */);
    // node ranges
    ostrm << "# i_zone i_node_head i_node_tail\n";
    for (int z = 1; z <= n_zones; ++z) {
      auto [head, tail] = part_to_nodes[p][z];
      if (head) {
        ostrm << z << ' ' << head << ' ' << tail << '\n';
      }
    }
    // send nodes info
    ostrm << "# i_part i_node_metis\n";
    for (auto& [recv_pid, nodes] : sendp_recvp_nodes[p]) {
      for (auto i : nodes) {
        ostrm << recv_pid << ' ' << i << '\n';
      }
    }
    // adjacent nodes
    ostrm << "# i_part i_node_metis i_zone i_node\n";
    for (auto& [i_part, nodes] : part_adj_nodes[p]) {
      for (auto mid : nodes) {
        auto& info = mapper.metis_to_cgns_for_nodes[mid];
        int zid = info.i_zone, nid = info.i_node;
        ostrm << i_part << ' ' << mid << ' ' << zid << ' ' << nid << '\n';
      }
    }
    // cell ranges
    ostrm << "# i_zone i_sect i_cell_head i_cell_tail\n";
    for (int z = 1; z <= n_zones; ++z) {
      auto n_sects = part_to_cells[p][z].size() - 1;
      for (int s = 1; s <= n_sects; ++s) {
        auto [head, tail] = part_to_cells[p][z][s];
        if (head) {
          ostrm << z << ' ' << s << ' ' << head << ' ' << tail << '\n';
        }
      }
    }
    // inner adjacency
    ostrm << "# i_cell_metis j_cell_metis\n";
    for (auto [i, j] : inner_adjs[p]) {
      ostrm << i << ' ' << j << '\n';
    }
    // interpart adjacency
    ostrm << "# i_part i_cell_metis j_cell_metis i_node_cnt j_node_cnt\n";
    for (auto& [i_part, pairs] : part_interpart_adjs[p]) {
      for (auto [i, j] : pairs) {
        auto& info_i = mapper.metis_to_cgns_for_cells[i];
        auto& info_j = mapper.metis_to_cgns_for_cells[j];
        int cnt_i = base.GetZone(info_i.i_zone).GetSection(info_i.i_sect).
            CountNodesByType();
        int cnt_j = base.GetZone(info_j.i_zone).GetSection(info_j.i_sect).
            CountNodesByType();
        ostrm << i_part << ' ' << i << ' ' << j << ' ' << cnt_i << ' ' <<
            cnt_j << '\n';
      }
    }
    ostrm << "#\n";
  }
}

}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  if (argc > 1) {
    mini::mesh::n_parts = std::atoi(argv[1]);
  }
  return RUN_ALL_TESTS();
}
