// Copyright 2019 Weicheng Pei and Minghao Yang

#include <cassert>
#include <cstdio>
#include <cstdlib>
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
#include "mini/data/path.hpp"  // defines TEST_DATA_DIR

namespace mini {
namespace mesh {

class ShufflerTest : public ::testing::Test {
 protected:
  using CgnsMesh = mini::mesh::cgns::File<double>;
  using MetisMesh = metis::Mesh<idx_t>;
  using MapperType = mini::mesh::mapper::CgnsToMetis<double, idx_t>;
  using FieldType = mini::mesh::cgns::Field<double>;
  std::string const test_data_dir_{TEST_DATA_DIR};
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
  for (int i_zone = 1; i_zone <= n_zones; ++i_zone) {
    auto& zone = base.GetZone(i_zone);
    // write data on nodes
    auto& node_sol = zone.AddSolution("NodeData",
        CGNS_ENUMV(Vertex));
    node_sol.AddField("NodePart");
    node_sol.AddField("MetisNodeId");
    auto& node_field = node_sol.GetField(1);
    auto& metis_nids = node_sol.GetField(2);
    auto n_nodes = zone.CountNodes();
    for (int cgns_nid = 1; cgns_nid <= n_nodes; ++cgns_nid) {
      auto metis_nid = zone_to_nodes[i_zone][cgns_nid];
      node_field.at(cgns_nid) = node_parts[metis_nid];
      metis_nids.at(cgns_nid) = metis_nid;
    }
    // write data on cells
    auto& cell_sol =  zone.AddSolution("CellData",
        CGNS_ENUMV(CellCenter));
    cell_sol.AddField("CellPart");
    cell_sol.AddField("MetisCellId");
    auto& cell_field = cell_sol.GetField(1);
    auto& metis_cids = cell_sol.GetField(2);
    auto& sect_to_cells = zone_to_sects.at(i_zone);
    auto n_sects = zone.CountSections();
    for (int i_sect = 1; i_sect <= n_sects; ++i_sect) {
      auto& sect = zone.GetSection(i_sect);
      if (sect.dim() != base.GetCellDim())
        continue;
      auto& cells_local_to_global = sect_to_cells.at(i_sect);
      auto n_cells = sect.CountCells();
      auto range_min{sect.CellIdMin()};
      auto range_max{sect.CellIdMax()};
      EXPECT_EQ(n_cells - 1, range_max - range_min);
      for (int cgns_i_cell = range_min; cgns_i_cell <= range_max;
           ++cgns_i_cell) {
        auto metis_i_cell = cells_local_to_global.at(cgns_i_cell);
        cell_field.at(cgns_i_cell) = cell_parts[metis_i_cell];
        metis_cids.at(cgns_i_cell) = metis_i_cell;
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
  MapperType mapper;
  using MeshDataType = double;
  using MetisId = idx_t;
  auto old_file_name = "hexa_old.cgns";
  auto new_file_name = "hexa_new.cgns";
  auto gmsh_cmd = std::string("gmsh ");
  gmsh_cmd += test_data_dir_;
  gmsh_cmd += "/double_mach_hexa.geo -save -o ";
  gmsh_cmd += old_file_name;
  std::system(gmsh_cmd.c_str());
  auto cgns_mesh = CgnsMesh(old_file_name);
  cgns_mesh.ReadBases();
  auto metis_mesh = mapper.Map(cgns_mesh);
  EXPECT_TRUE(mapper.IsValid());
  MetisId n_parts{4}, n_common_nodes{3};
  auto graph = metis::MeshToDual(metis_mesh, n_common_nodes);
  auto cell_parts = metis::PartGraph(graph, n_parts);
  std::vector<idx_t> node_parts = metis::GetNodeParts(
      metis_mesh, cell_parts, n_parts);
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
  // auto [begin_nid, end_nid] = part_to_nodes[i_part][i_zone];
  // auto [begin_cid, end_cid] = part_to_cells[i_part][i_zone][i_sect];
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
    auto& node_field = zone.GetSolution(1).GetField(1);
    assert(node_field.name() == "NodePart");
    // slice node lists by i_part
    int prev_nid = 1, prev_part = node_field.at(prev_nid);
    int n_nodes = zone.CountNodes();
    for (int curr_nid = prev_nid+1; curr_nid <= n_nodes; ++curr_nid) {
      int curr_part = node_field.at(curr_nid);
      if (curr_part != prev_part) {
        part_to_nodes[prev_part][zid] = std::make_pair(prev_nid, curr_nid);
        prev_nid = curr_nid;
        prev_part = curr_part;
      }
    }
    part_to_nodes[prev_part][zid] = std::make_pair(prev_nid, n_nodes+1);
    // slice cell lists by i_part
    int n_cells = zone.CountCells();
    auto& sol = zone.GetSolution(2);
    assert(sol.name() == "CellData");
    auto& cell_field = sol.GetField(1);
    assert(cell_field.name() == "CellPart");
    for (int sid = 1; sid <= zone.CountSections(); ++sid) {
      auto& sect = zone.GetSection(sid);
      if (sect.dim() != base.GetCellDim())
        continue;
      int cid_min = sect.CellIdMin(), cid_max = sect.CellIdMax();
      int prev_cid = cid_min, prev_part = cell_field.at(prev_cid);
      for (int curr_cid = prev_cid+1; curr_cid <= cid_max; ++curr_cid) {
        int curr_part = cell_field.at(curr_cid);
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
  // store cell adjacency for each part
  // inner_adjs[i_part] = vector of [smaller_cid, bigger_cid]
  auto inner_adjs = std::vector<std::vector<std::pair<int, int>>>(n_parts);
  auto part_interpart_adjs = std::vector<std::map<int, std::vector<
      std::pair<int, int>>>>(n_parts);
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
  // write to txts
  for (int p = 0; p < n_parts; ++p) {
    auto ostrm = std::ofstream("hexa_part_" + std::to_string(p) + ".txt"
        /* , std::ios::binary */);
    // node ranges
    for (int z = 1; z <= n_zones; ++z) {
      auto [head, tail] = part_to_nodes[p][z];
      if (head) {
        ostrm << z << ' ' << head << ' ' << tail << '\n';
      }
    }
    ostrm << '\n';
    // send nodes info
    for (auto& [recv_pid, nodes] : sendp_recvp_nodes[p]) {
      for (auto i : nodes) {
        ostrm << recv_pid << ' ' << i << '\n';
      }
    }
    ostrm << '\n';
    // adjacent nodes
    for (auto& [i_part, nodes] : part_adj_nodes[p]) {
      for (auto mid : nodes) {
        auto& info = mapper.metis_to_cgns_for_nodes[mid];
        int zid = info.i_zone, nid = info.i_node;
        ostrm << i_part << ' ' << mid << ' ' << zid << ' ' << nid << '\n';
      }
    }
    ostrm << '\n';
    // cell ranges
    for (int z = 1; z <= n_zones; ++z) {
      auto n_sects = part_to_cells[p][z].size() - 1;
      for (int s = 1; s <= n_sects; ++s) {
        auto [head, tail] = part_to_cells[p][z][s];
        if (head) {
          ostrm << z << ' ' << s << ' ' << head << ' ' << tail << '\n';
        }
      }
    }
    ostrm << '\n';
    // inner adjacency
    for (auto [i, j] : inner_adjs[p]) {
      ostrm << i << ' ' << j << '\n';
    }
    ostrm << '\n';
    // interpart adjacency
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
  }
}

}  // namespace mesh
}  // namespace mini

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
