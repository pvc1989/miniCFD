// Copyright 2020 PEI Weicheng and YANG Minghao

#ifndef MINI_MESH_MAPPER_HPP_
#define MINI_MESH_MAPPER_HPP_

#include <concepts>

#include <cassert>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "cgnslib.h"

#include "mini/mesh/cgns.hpp"
#include "mini/mesh/metis.hpp"

namespace mini {
namespace mesh {
namespace mapper {

static_assert(sizeof(idx_t) == sizeof(cgsize_t),
    "METIS's `idx_t` and CGNS's `cgsize_t` must have the same size.");

template <std::integral Int, std::floating_point Real>
struct CgnsToMetis {
  using CgnsMesh = cgns::File<Real>;
  using MetisMesh = metis::Mesh<Int>;
  using ShiftedVector = cgns::ShiftedVector<Int>;

  MetisMesh Map(CgnsMesh const &mesh);
  bool IsValid() const;
  void WriteParts(
      std::vector<idx_t> const &cell_parts,
      std::vector<idx_t> const &node_parts, CgnsMesh *cgns_mesh) const;
  void WriteToFile(const std::string &name) const {
    auto ostrm = std::ofstream(name);
    Int metis_n_nodes = metis_to_cgns_for_nodes.size();
    for (Int metis_i_node = 0; metis_i_node < metis_n_nodes; ++metis_i_node) {
      auto index = metis_to_cgns_for_nodes.at(metis_i_node);
      int zid = index.i_zone, nid = index.i_node;
      ostrm << metis_i_node << ' ' << zid << ' ' << nid << '\n';
    }
  }

  std::vector<cgns::NodeIndex<Int>> metis_to_cgns_for_nodes;
  std::vector<cgns::CellIndex<Int>> metis_to_cgns_for_cells;
  // metis_i_node =
  //     cgns_to_metis_for_nodes[i_zone][i_node];
  std::vector<std::vector<Int>>           cgns_to_metis_for_nodes;
  // metis_i_cell =
  //     cgns_to_metis_for_cells[i_zone][i_sect][i_cell];
  std::vector<std::vector<ShiftedVector>> cgns_to_metis_for_cells;
};

template <std::integral Int, std::floating_point Real>
typename CgnsToMetis<Int, Real>::MetisMesh
CgnsToMetis<Int, Real>::Map(CgnsMesh const &cgns_mesh) {
  assert(cgns_mesh.CountBases() == 1);
  auto &base = cgns_mesh.GetBase(1);
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
  for (Int i_zone = 1; i_zone <= n_zones; ++i_zone) {
    auto &zone = base.GetZone(i_zone);
    // read nodes in current zone
    auto n_nodes_in_curr_zone = zone.CountNodes();
    auto &nodes = cgns_to_metis_for_nodes.at(i_zone);
    nodes.reserve(n_nodes_in_curr_zone+1);
    nodes.emplace_back(-1);  // nodes[0] is invalid
    metis_to_cgns_for_nodes.reserve(metis_to_cgns_for_nodes.size() +
                                    n_nodes_in_curr_zone);
    for (Int i_node = 1; i_node <= n_nodes_in_curr_zone; ++i_node) {
      metis_to_cgns_for_nodes.emplace_back(i_zone, i_node);
      nodes.emplace_back(n_nodes_in_curr_base++);
    }
    // read cells in current zone
    auto n_sections = zone.CountSections();
    cgns_to_metis_for_cells[i_zone].resize(n_sections + 1);
    for (Int i_sect = 1; i_sect <= n_sections; ++i_sect) {
      auto &section = zone.GetSection(i_sect);
      if (!zone.CheckTypeDim(section.type(), cell_dim))
        continue;
      auto &metis_i_cells = cgns_to_metis_for_cells[i_zone][i_sect];
      metis_i_cells = ShiftedVector(section.CountCells(), section.CellIdMin());
      auto n_cells_in_curr_sect = section.CountCells();
      auto n_nodes_per_cell = section.CountNodesByType();
      metis_to_cgns_for_cells.reserve(metis_to_cgns_for_cells.size() +
                                      n_cells_in_curr_sect);
      cell_ptr.reserve(cell_ptr.size() + n_cells_in_curr_sect);
      for (Int i_cell = section.CellIdMin();
           i_cell <= section.CellIdMax(); ++i_cell) {
        metis_i_cells.at(i_cell) = metis_to_cgns_for_cells.size();
        metis_to_cgns_for_cells.emplace_back(i_zone, i_sect, i_cell);
        cell_ptr.emplace_back(pointer_value += n_nodes_per_cell);
      }
      auto i_node_list_size = n_nodes_per_cell * n_cells_in_curr_sect;
      cell_idx.reserve(cell_idx.size() + i_node_list_size);
      auto i_node_list = section.GetNodeIdList();
      for (Int i_node = 0; i_node < i_node_list_size; ++i_node) {
        auto metis_i_node = n_nodes_in_prev_zones + i_node_list[i_node] - 1;
        cell_idx.emplace_back(metis_i_node);
      }
    }
    n_nodes_in_prev_zones += zone.CountNodes();
  }
  assert(metis_to_cgns_for_nodes.size() == n_nodes_in_curr_base);
  return MetisMesh(std::move(cell_ptr), std::move(cell_idx),
      n_nodes_in_curr_base);
}

template <std::integral Int, std::floating_point Real>
bool CgnsToMetis<Int, Real>::IsValid() const {
  Int metis_n_nodes = metis_to_cgns_for_nodes.size();
  for (Int metis_i_node = 0; metis_i_node < metis_n_nodes; ++metis_i_node) {
    auto index = metis_to_cgns_for_nodes.at(metis_i_node);
    auto &nodes = cgns_to_metis_for_nodes.at(index.i_zone);
    if (nodes.at(index.i_node) != metis_i_node) {
      return false;
    }
  }
  Int metis_n_cells = metis_to_cgns_for_cells.size();
  for (Int metis_i_cell = 0; metis_i_cell < metis_n_cells; ++metis_i_cell) {
    auto index = metis_to_cgns_for_cells.at(metis_i_cell);
    auto &cells = cgns_to_metis_for_cells.at(index.i_zone).at(index.i_sect);
    if (cells.at(index.i_cell) != metis_i_cell) {
      return false;
    }
  }
  return true;
}

template <std::integral Int, std::floating_point Real>
void CgnsToMetis<Int, Real>::WriteParts(
    std::vector<idx_t> const &cell_parts,
    std::vector<idx_t> const &node_parts, CgnsMesh *cgns_mesh) const {
  auto &zone_to_sects = cgns_to_metis_for_cells;
  auto &zone_to_nodes = cgns_to_metis_for_nodes;
  auto &base = cgns_mesh->GetBase(1);
  int n_zones = base.CountZones();
  for (int i_zone = 1; i_zone <= n_zones; ++i_zone) {
    auto &zone = base.GetZone(i_zone);
    // write data on nodes
    auto &node_sol = zone.AddSolution("DataOnNodes", CGNS_ENUMV(Vertex));
    auto &node_field = node_sol.AddField("PartIndex");
    auto &metis_nids = node_sol.AddField("MetisIndex");
    auto n_nodes = zone.CountNodes();
    for (int cgns_nid = 1; cgns_nid <= n_nodes; ++cgns_nid) {
      auto metis_nid = zone_to_nodes[i_zone][cgns_nid];
      node_field.at(cgns_nid) = node_parts[metis_nid];
      metis_nids.at(cgns_nid) = metis_nid;
    }
    // write data on cells
    auto &cell_sol =  zone.AddSolution("DataOnCells", CGNS_ENUMV(CellCenter));
    auto &cell_field = cell_sol.AddField("PartIndex");
    auto &metis_cids = cell_sol.AddField("MetisIndex");
    auto &sect_to_cells = zone_to_sects.at(i_zone);
    auto n_sects = zone.CountSections();
    for (int i_sect = 1; i_sect <= n_sects; ++i_sect) {
      auto &sect = zone.GetSection(i_sect);
      if (sect.dim() != base.GetCellDim())
        continue;
      auto &cells_local_to_global = sect_to_cells.at(i_sect);
      auto n_cells = sect.CountCells();
      auto range_min{sect.CellIdMin()};
      auto range_max{sect.CellIdMax()};
      assert(n_cells - 1 == range_max - range_min);
      for (int cgns_i_cell = range_min; cgns_i_cell <= range_max;
           ++cgns_i_cell) {
        auto metis_i_cell = cells_local_to_global.at(cgns_i_cell);
        cell_field.at(cgns_i_cell) = cell_parts[metis_i_cell];
        metis_cids.at(cgns_i_cell) = metis_i_cell;
      }
    }
  }
}

}  // namespace mapper
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_MAPPER_HPP_
