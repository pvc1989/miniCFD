// Copyright 2020 Weicheng Pei and Minghao Yang

#ifndef MINI_MESH_CGNS_CONVERTER_HPP_
#define MINI_MESH_CGNS_CONVERTER_HPP_

#include <cassert>
#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "cgnslib.h"

#include "mini/mesh/cgns/format.hpp"

namespace mini {
namespace mesh {
namespace cgns {

struct NodeInfo {
  NodeInfo(int zi, int ni) : zone_id(zi), node_id(ni) {}
  int zone_id{0};
  int node_id{0};
};
struct CellInfo {
  CellInfo(int zi, int si, int ci) : zone_id(zi), section_id(si), cell_id(ci) {}
  int zone_id{0};
  int section_id{0};
  int cell_id{0};
};

template <typename CgnsMesh, typename MetisMesh>
struct Converter {
  Converter() = default;
  MetisMesh ConvertToMetisMesh(const CgnsMesh& mesh);
  std::vector<NodeInfo> metis_to_cgns_for_nodes;
  std::map<int, std::vector<int>> cgns_to_metis_for_nodes;
  std::vector<CellInfo> metis_to_cgns_for_cells;
  std::map<int, std::map<int, std::vector<int>>> cgns_to_metis_for_cells;
};
template <typename CgnsMesh, typename MetisMesh>
MetisMesh Converter<CgnsMesh, MetisMesh>::ConvertToMetisMesh(
    const CgnsMesh& cgns_mesh) {
  assert(cgns_mesh.CountBases() == 1);
  auto metis_mesh = MetisMesh();
  auto& base = cgns_mesh.GetBase(1);
  auto cell_dim = base.GetCellDim();
  auto n_zones = base.CountZones();
  int n_nodes_of_curr_base{0};
  auto& cell_ptr = metis_mesh.cells.range;
  auto& cell_idx = metis_mesh.cells.index;
  int pointer_value{0};
  cell_ptr.emplace_back(pointer_value);
  auto n_nodes_in_prev_zones{0};
  for (int zone_id = 1; zone_id <= n_zones; ++zone_id) {
    auto& zone = base.GetZone(zone_id);
    // read nodes in current zone
    auto n_nodes_of_curr_zone = zone.CountNodes();
    cgns_to_metis_for_nodes.emplace(zone_id, std::vector<int>());
    auto& nodes = cgns_to_metis_for_nodes.at(zone_id);
    nodes.reserve(n_nodes_of_curr_zone+1);
    nodes.emplace_back(-1);
    metis_to_cgns_for_nodes.reserve(metis_to_cgns_for_nodes.size() +
                                      n_nodes_of_curr_zone);
    for (int node_id = 1; node_id <= n_nodes_of_curr_zone; ++node_id) {
      metis_to_cgns_for_nodes.emplace_back(zone_id, node_id);
      nodes.emplace_back(n_nodes_of_curr_base++);
    }
    // read cells in current zone
    cgns_to_metis_for_cells.emplace(zone_id, std::map<int, std::vector<int>>());
    auto n_sections = zone.CountSections();
    for (int section_id = 1; section_id <= n_sections; ++section_id) {
      auto [iter, succ] = cgns_to_metis_for_cells[zone_id].emplace(
          section_id, std::vector<int>());
      auto& metis_ids_in_section = iter->second;
      auto& section = zone.GetSection(section_id);
      if (!CheckTypeDim(section.type(), cell_dim)) continue;
      auto n_cells_of_curr_sect = section.CountCells();

      auto n_nodes_per_cell = section.CountNodesByType(section.type());
      metis_to_cgns_for_cells.reserve(metis_to_cgns_for_cells.size() +
                                      n_cells_of_curr_sect);
      cell_ptr.reserve(cell_ptr.size() + n_cells_of_curr_sect);
      for (int cell_id = section.CellIdMin();
           cell_id <= section.CellIdMax(); ++cell_id) {
        metis_ids_in_section.emplace_back(metis_to_cgns_for_cells.size());
        metis_to_cgns_for_cells.emplace_back(zone_id, section_id, cell_id);
        cell_ptr.emplace_back(pointer_value+=n_nodes_per_cell);
      }
      auto node_id_list_size = n_nodes_per_cell * n_cells_of_curr_sect;
      cell_idx.reserve(cell_idx.size() + node_id_list_size);
      auto node_id_list = section.GetNodeIdList();
      for (int node_id = 0; node_id < node_id_list_size; ++node_id) {
        auto node_id_global = n_nodes_in_prev_zones + node_id_list[node_id] - 1;
        cell_idx.emplace_back(node_id_global);
      }
    }
    n_nodes_in_prev_zones += zone.CountNodes();
  }
  return metis_mesh;
}

}  // namespace cgns
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_CGNS_CONVERTER_HPP_
