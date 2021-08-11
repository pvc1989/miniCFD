// Copyright 2020 Weicheng Pei and Minghao Yang

#ifndef MINI_MESH_CGNS_SHUFFLER_HPP_
#define MINI_MESH_CGNS_SHUFFLER_HPP_

#include <cassert>
#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "cgnslib.h"

#include "mini/mesh/metis/format.hpp"
#include "mini/mesh/filter/cgns_to_metis.hpp"

namespace mini {
namespace mesh {

template <typename T>
std::vector<T> GetNodePartsByConnectivity(
    const metis::Mesh<int>& mesh, const std::vector<T>& cell_parts,
    T n_parts, int n_nodes) {
  auto node_parts = std::vector<T>(n_nodes, n_parts);
  auto node_pointer = &(mesh.index(0));
  int n_cells = mesh.CountCells();
  auto curr_range_pointer = &(mesh.nodes(0));
  for (int i = 0; i < n_cells; ++i) {
    auto part_value = cell_parts[i];
    auto head = *curr_range_pointer++;
    auto tail = *curr_range_pointer;
    for (int j = head; j < tail; ++j) {
      int node_index = *node_pointer++;
      if (part_value < node_parts[node_index])
        node_parts[node_index] = part_value;
    }
  }
  return node_parts;
}

template <typename T>
void ReorderByParts(const std::vector<T>& parts, int* new_order) {
  std::map<int, std::set<int>> collector;
  int n = parts.size();
  for (int local_id = 0; local_id < n; ++local_id) {
    int part = parts[local_id];
    if (collector.find(part) != collector.end()) {
      collector[part].insert(local_id);
    } else {
      collector.emplace(part, std::set<int>());
      collector[part].insert(local_id);
    }
  }
  int index = 0;
  for (auto& part : collector) {
    for (auto& local_id : part.second) {
      *new_order++ = local_id;
    }
  }
}
template <typename T>
void ShuffleConnectivity(const std::vector<int>& new_cell_order, int npe,
                         T* node_id_list) {
  int n_cells = new_cell_order.size();
  int node_id_list_size = n_cells * npe;
  std::vector<T> new_node_id_list(node_id_list_size);
  auto new_ptr = new_node_id_list.data();
  for (int i = 0; i < n_cells; ++i) {
    int range_min = npe * new_cell_order[i];
    auto old_ptr = node_id_list + range_min;
    for (int j = range_min; j < range_min + npe; ++j) {
      *new_ptr++ = *old_ptr++;
    }
  }
  auto old_ptr = node_id_list;
  for (int i = 0; i < node_id_list_size; ++i) {
    *old_ptr++ = new_node_id_list[i];
  }
}
template <typename T>
void ShuffleDataArray(const std::vector<int>& new_cell_order, T* cell_data) {
  int n = new_cell_order.size();
  std::vector<T> new_cell_data(n);
  for (int i = 0; i < n; ++i) {
    new_cell_data[i] = *(cell_data + new_cell_order[i]);
  }
  auto cell_data_ptr = cell_data;
  for (int i = 0; i < n; ++i) {
    *cell_data_ptr++ = new_cell_data[i];
  }
}

template <typename T, class Real>
class Shuffler {
 public:
  using CgnsFile = mini::mesh::cgns::File<Real>;
  using MetisMesh = metis::Mesh<int>;
  using FilterType = mini::mesh::filter::CgnsToMetis<double, int>;
  using SectionType = mini::mesh::cgns::Section<Real>;
  using SolutionType = mini::mesh::cgns::Solution<Real>;
  using FieldType = mini::mesh::cgns::Field<Real>;
  Shuffler() = default;
  void SetNumParts(int n_parts) {
    n_parts_ = n_parts;
  }
  void SetCellParts(std::vector<T>* cell_parts) {
    cell_parts_ = cell_parts;
  }
  void SetMetisMesh(MetisMesh* mesh) {
    mesh_ = mesh;
  }
  void SetFilter(FilterType* filter) {
    filter_ = filter;
  }
  void ShuffleMesh(CgnsFile* mesh) {
    auto& zone_to_sections = filter_->cgns_to_metis_for_cells;
    auto& base = mesh->GetBase(1); int n_zones = base.CountZones();
    for (int zone_id = 1; zone_id <= n_zones; ++zone_id) {
      auto& zone = base.GetZone(zone_id);
      auto& section_to_cells = zone_to_sections.at(zone_id);
      int n_sections = zone.CountSections();
      for (int section_id = 1; section_id <= n_sections; ++section_id) {
        auto& section = zone.GetSection(section_id);
        if (section.dim() != base.GetCellDim())
          continue;
        auto& cells_local_to_global = section_to_cells.at(section_id);
        int n_cells = section.CountCells();
        std::vector<int> parts(n_cells);
        for (int local_id = 0; local_id < n_cells; ++local_id) {
          auto global_id = cells_local_to_global[local_id];
          parts[local_id] = (*cell_parts_)[global_id];
        }
        int range_min{section.CellIdMin()-1};
            /* Shuffle Connectivity */
        std::vector<int> new_order(n_cells);
        ReorderByParts<int>(parts, new_order.data());
        int npe = section.CountNodesByType();
        cgsize_t* node_id_list = section.GetNodeIdList();
        ShuffleConnectivity<cgsize_t>(new_order, npe, node_id_list);
            /* Shuffle CellData */
        int n_solutions = zone.CountSolutions();
        for (int solution_id = 1; solution_id <= n_solutions; solution_id++) {
          auto& solution = zone.GetSolution(solution_id);
          for (auto& [name, field] : solution.fields()) {
            auto field_ptr = field.data() + range_min;
            ShuffleDataArray<double>(new_order, field_ptr);
          }
        }
      }
    }
  }

 private:
  int n_parts_;
  std::vector<T>* cell_parts_;
  MetisMesh* mesh_;
  FilterType* filter_;
};

}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_CGNS_SHUFFLER_HPP_
