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

#include "mini/mesh/cgns/tree.hpp"

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

struct CompressedSparseRowMatrix {
  std::vector<int> pointer;
  std::vector<int> index;
};

struct MetisMesh {
  CompressedSparseRowMatrix csr_matrix_for_cells;
};

struct Converter {
  using CgneMesh = Tree<double>;
  Converter() = default;
  std::unique_ptr<MetisMesh> ConvertToMetisMesh(const CgneMesh* mesh);
  std::vector<NodeInfo> global_to_local_for_nodes;
  std::map<int, std::vector<int>> local_to_global_for_nodes;
  std::vector<CellInfo> global_to_local_for_cells;
};

std::unique_ptr<MetisMesh> Converter::ConvertToMetisMesh(
    const Converter::CgneMesh* cgns_mesh) {
  assert(cgns_mesh->CountBases() == 1);
  auto metis_mesh = std::make_unique<MetisMesh>();
  auto& base = cgns_mesh->GetBase(1);
  auto cell_dim = base.GetCellDim();
  std::set<CGNS_ENUMT(ElementType_t)> types;
  if (cell_dim == 2) {
    types.insert(CGNS_ENUMV(TRI_3));
    types.insert(CGNS_ENUMV(QUAD_4));
  }
  else if (cell_dim == 3) {
    types.insert(CGNS_ENUMV(TETRA_4));
    types.insert(CGNS_ENUMV(HEXA_8));
  }
  auto n_zones = base.CountZones();
  int n_nodes_of_curr_base{0};
  auto& cell_ptr = metis_mesh->csr_matrix_for_cells.pointer;
  auto& cell_ind = metis_mesh->csr_matrix_for_cells.index;
  int pointer_value{0};
  cell_ptr.emplace_back(pointer_value);
  auto n_nodes_in_prev_zones{0};
  for (int zone_id = 1; zone_id <= n_zones; ++zone_id) {
    auto& zone = base.GetZone(zone_id);
    // read nodes in current zone
    auto n_nodes_of_curr_zone = zone.CountNodes();
    local_to_global_for_nodes.emplace(zone_id, std::vector<int>());
    auto& nodes = local_to_global_for_nodes.at(zone_id);
    nodes.reserve(n_nodes_of_curr_zone+1);
    nodes.emplace_back(-1);
    global_to_local_for_nodes.reserve(global_to_local_for_nodes.size() +
                                      n_nodes_of_curr_zone);
    for (int node_id = 1; node_id <= n_nodes_of_curr_zone; ++node_id) {
      global_to_local_for_nodes.emplace_back(zone_id, node_id);
      nodes.emplace_back(n_nodes_of_curr_base++);
    }
    // read cells in current zone
    auto n_sections = zone.CountSections();
    for (int section_id = 1; section_id <= n_sections; ++section_id) {
      auto& section = zone.GetSection(section_id);
      if (types.find(section.GetType()) == types.end()) continue;
      auto n_cells_of_curr_sect = section.CountCells();
      auto n_nodes_per_cell = CountNodesByType(section.GetType());
      global_to_local_for_cells.reserve(global_to_local_for_cells.size() +
                                        n_cells_of_curr_sect);
      cell_ptr.reserve(cell_ptr.size() + n_cells_of_curr_sect);
      for (int cell_id = section.GetOneBasedCellIdMin();
           cell_id <= section.GetOneBasedCellIdMax(); ++cell_id) {
        global_to_local_for_cells.emplace_back(zone_id, section_id, cell_id);
        cell_ptr.emplace_back(pointer_value+=n_nodes_per_cell);
      }
      auto connectivity_size = n_nodes_per_cell * n_cells_of_curr_sect;
      cell_ind.reserve(cell_ind.size() + connectivity_size);
      auto connectivity = section.GetConnectivity();
      for (int node_id = 0; node_id < connectivity_size; ++node_id) {
        auto node_id_global = n_nodes_in_prev_zones + connectivity[node_id] - 1;
        cell_ind.emplace_back(node_id_global);
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