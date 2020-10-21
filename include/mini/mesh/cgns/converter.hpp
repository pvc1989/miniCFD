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

template <typename T>
struct CompressedSparseRowMatrix {
  std::vector<T> pointer;
  std::vector<T> index;
};

template <typename T>
struct MetisMesh {
  CompressedSparseRowMatrix<T> csr_matrix_for_cells;
};

template <typename T>
struct Converter {
  using CgneMesh = Tree<double>;
  Converter() = default;
  std::unique_ptr<MetisMesh<T>> ConvertToMetisMesh(const CgneMesh* mesh);
  std::vector<NodeInfo> metis_to_cgns_for_nodes;
  std::map<int, std::vector<int>> cgns_to_metis_for_nodes;
  std::vector<CellInfo> metis_to_cgns_for_cells;
  std::map<int, std::map<int, std::vector<int>>> cgns_to_metis_for_cells;
};
template <typename T>
std::unique_ptr<MetisMesh<T>> Converter<T>::ConvertToMetisMesh(
    const Converter<T>::CgneMesh* cgns_mesh) {
  assert(cgns_mesh->CountBases() == 1);
  auto metis_mesh = std::make_unique<MetisMesh<T>>();
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
      auto [iter, succ] = cgns_to_metis_for_cells[zone_id].emplace(section_id, std::vector<int>());
      auto& metis_ids_in_section = iter->second;
      auto& section = zone.GetSection(section_id);
      if (types.find(section.GetType()) == types.end()) continue;
      auto n_cells_of_curr_sect = section.CountCells();

      auto n_nodes_per_cell = CountNodesByType(section.GetType());
      metis_to_cgns_for_cells.reserve(metis_to_cgns_for_cells.size() +
                                      n_cells_of_curr_sect);
      cell_ptr.reserve(cell_ptr.size() + n_cells_of_curr_sect);
      for (int cell_id = section.GetOneBasedCellIdMin();
           cell_id <= section.GetOneBasedCellIdMax(); ++cell_id) {
        metis_ids_in_section.emplace_back(metis_to_cgns_for_cells.size());
        metis_to_cgns_for_cells.emplace_back(zone_id, section_id, cell_id);
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
