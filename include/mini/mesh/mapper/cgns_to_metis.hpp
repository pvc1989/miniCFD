// Copyright 2020 Weicheng Pei and Minghao Yang

#ifndef MINI_MESH_MAPPER_CGNS_TO_METIS_HPP_
#define MINI_MESH_MAPPER_CGNS_TO_METIS_HPP_

#include <cassert>
#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

#include "cgnslib.h"

#include "mini/mesh/cgns/format.hpp"
#include "mini/mesh/metis/format.hpp"

namespace mini {
namespace mesh {
namespace mapper {

static_assert(std::is_same_v<idx_t, cgsize_t>,
    "METIS's `idx_t` is different from CGNS's `cgsize_t`.");

template <class Int = int>
struct NodeInfo {
  NodeInfo() = default;
  NodeInfo(Int zi, Int ni) : zone_id(zi), node_id(ni) {}
  Int zone_id{0}, node_id{0};
};
template <class Int = int>
struct CellInfo {
  CellInfo() = default;
  CellInfo(Int zi, Int si, Int ci) : zone_id(zi), section_id(si), cell_id(ci) {}
  Int zone_id{0}, section_id{0}, cell_id{0};
};

template <class Real = double, class Int = int>
struct CgnsToMetis {
  using CgnsMesh = cgns::File<Real>;
  using MetisMesh = metis::Mesh<Int>;

  MetisMesh Map(const CgnsMesh& mesh);

  std::vector<NodeInfo<Int>> metis_to_cgns_for_nodes;
  std::vector<CellInfo<Int>> metis_to_cgns_for_cells;
  // metis_node_id =
  //     cgns_to_metis_for_nodes[zone_id][node_id];
  std::vector<std::vector<Int>>              cgns_to_metis_for_nodes;
  // metis_cell_id =
  //     cgns_to_metis_for_cells[zone_id][sect_id][cell_id - cell_id_min];
  std::vector<std::vector<std::vector<Int>>> cgns_to_metis_for_cells;
};
template <class Real, class Int>
typename CgnsToMetis<Real, Int>::MetisMesh
CgnsToMetis<Real, Int>::Map(const CgnsMesh& cgns_mesh) {
  assert(cgns_mesh.CountBases() == 1);
  auto& base = cgns_mesh.GetBase(1);
  auto cell_dim = base.GetCellDim();
  auto n_zones = base.CountZones();
  Int n_nodes_in_curr_base{0};
  auto cell_ptr = std::vector<Int>();  // METIS node range
  auto cell_idx = std::vector<Int>();  // METIS node index
  Int pointer_value{0};
  cell_ptr.emplace_back(pointer_value);
  Int n_nodes_in_prev_zones{0};
  cgns_to_metis_for_nodes.resize(n_zones+1);
  cgns_to_metis_for_cells.resize(n_zones+1);
  for (Int zone_id = 1; zone_id <= n_zones; ++zone_id) {
    auto& zone = base.GetZone(zone_id);
    // read nodes in current zone
    auto n_nodes_in_curr_zone = zone.CountNodes();
    auto& nodes = cgns_to_metis_for_nodes.at(zone_id);
    nodes.reserve(n_nodes_in_curr_zone+1);
    nodes.emplace_back(-1);  // nodes[0] is invalid
    metis_to_cgns_for_nodes.reserve(metis_to_cgns_for_nodes.size() +
                                    n_nodes_in_curr_zone);
    for (Int node_id = 1; node_id <= n_nodes_in_curr_zone; ++node_id) {
      metis_to_cgns_for_nodes.emplace_back(zone_id, node_id);
      nodes.emplace_back(n_nodes_in_curr_base++);
    }
    // read cells in current zone
    auto n_sections = zone.CountSections();
    cgns_to_metis_for_cells[zone_id].resize(n_sections + 1);
    for (Int section_id = 1; section_id <= n_sections; ++section_id) {
      auto& section = zone.GetSection(section_id);
      if (!zone.CheckTypeDim(section.type(), cell_dim))
        continue;
      auto& metis_ids_in_section = cgns_to_metis_for_cells[zone_id][section_id];
      // metis_ids_in_section.emplace_back(-1);  // cells[0] is invalid
      auto n_cells_in_curr_sect = section.CountCells();
      auto n_nodes_per_cell = section.CountNodesByType();
      metis_to_cgns_for_cells.reserve(metis_to_cgns_for_cells.size() +
                                      n_cells_in_curr_sect);
      cell_ptr.reserve(cell_ptr.size() + n_cells_in_curr_sect);
      for (Int cell_id = section.CellIdMin();
           cell_id <= section.CellIdMax(); ++cell_id) {
        metis_ids_in_section.emplace_back(metis_to_cgns_for_cells.size());
        metis_to_cgns_for_cells.emplace_back(zone_id, section_id, cell_id);
        cell_ptr.emplace_back(pointer_value += n_nodes_per_cell);
      }
      auto node_id_list_size = n_nodes_per_cell * n_cells_in_curr_sect;
      cell_idx.reserve(cell_idx.size() + node_id_list_size);
      auto node_id_list = section.GetNodeIdList();
      for (Int node_id = 0; node_id < node_id_list_size; ++node_id) {
        auto metis_node_id = n_nodes_in_prev_zones + node_id_list[node_id] - 1;
        cell_idx.emplace_back(metis_node_id);
      }
    }
    n_nodes_in_prev_zones += zone.CountNodes();
  }
  assert(metis_to_cgns_for_nodes.size() == n_nodes_in_curr_base);
  return MetisMesh(cell_ptr, cell_idx, n_nodes_in_curr_base);
}

}  // namespace mapper
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_MAPPER_CGNS_TO_METIS_HPP_
