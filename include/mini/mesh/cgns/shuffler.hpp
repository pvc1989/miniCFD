// Copyright 2020 Weicheng Pei and Minghao Yang

#ifndef MINI_MESH_CGNS_SHUFFLER_HPP_
#define MINI_MESH_CGNS_SHUFFLER_HPP_

#include <cassert>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "cgnslib.h"

#include "mini/mesh/metis/format.hpp"
#include "mini/mesh/mapper/cgns_to_metis.hpp"

namespace mini {
namespace mesh {

template <typename T>
std::vector<T> GetNodePartsByConnectivity(
    const metis::Mesh<int>& mesh, const std::vector<T>& cell_parts,
    T n_parts, int n_nodes) {
  auto node_parts = std::vector<T>(n_nodes, n_parts);
  auto node_pointer = &(mesh.nodes(0));
  int n_cells = mesh.CountCells();
  auto curr_range_pointer = &(mesh.range(0));
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

/**
 * @brief Get the New Order object
 * 
 * @tparam T 
 * @param parts 
 * @param n 
 * @return std::vector<int> that maps `new_pos` to `old_pos`.
 */
template <typename T = int>
std::vector<int> GetNewOrder(const T* parts, int n) {
  auto new_order = std::vector<int>(n);
  std::iota(new_order.begin(), new_order.end(), 0);
  auto cmp = [parts](int lid, int rid) {
    return parts[lid] < parts[rid] || (parts[lid] == parts[rid] && lid < rid);
  };
  std::sort(new_order.begin(), new_order.end(), cmp);
  return new_order;
}
template <typename T>
void ShuffleData(const std::vector<int>& new_order, T* old_data) {
  int n = new_order.size();
  std::vector<T> new_data(n);
  for (int i = 0; i < n; ++i)
    new_data[i] = old_data[new_order[i]];
  std::memcpy(old_data, new_data.data(), n * sizeof(T));
  // for (int i = 0; i < n; ++i)
  //   old_data[i] = new_data[i];
}
template <typename T>
void ShuffleConnectivity(const std::vector<int>& new_to_old_for_nodes,
                         const std::vector<int>& new_to_old_for_cells,
                         int npe, T* old_cid_old_nid) {
  int n_cells = new_to_old_for_cells.size();
  int n_nodes = new_to_old_for_nodes.size();
  int node_id_list_size = n_cells * npe;
  auto old_to_new_for_nodes = new_to_old_for_nodes;
  for (int i = 0; i < n_nodes; ++i) {
    old_to_new_for_nodes.at(new_to_old_for_nodes.at(i)) = i;
  }
  std::vector<T> old_cid_new_nid(node_id_list_size);
  for (int i = 0; i < node_id_list_size; ++i) {
    auto old_nid = old_cid_old_nid[i];
    auto new_nid = old_to_new_for_nodes.at(old_nid - 1) + 1;
    old_cid_new_nid[i] = new_nid;
  }
  auto* new_cid_new_nid = old_cid_old_nid;
  for (int new_cid = 0; new_cid < n_cells; ++new_cid) {
    int range_min = npe * new_to_old_for_cells[new_cid];
    auto old_ptr = old_cid_new_nid.data() + range_min;
    for (int i = 0; i < npe; ++i) {
      *new_cid_new_nid++ = *old_ptr++;
    }
  }
}

template <typename T, class Real>
class Shuffler {
 public:
  using CgnsFile = mini::mesh::cgns::File<Real>;
  using MetisMesh = metis::Mesh<int>;
  using MapperType = mini::mesh::mapper::CgnsToMetis<double, int>;
  using SectionType = mini::mesh::cgns::Section<Real>;
  using SolutionType = mini::mesh::cgns::Solution<Real>;
  using FieldType = mini::mesh::cgns::Field<Real>;

  Shuffler(int n_parts, std::vector<T> const& cell_parts,
           std::vector<T> const& node_parts)
      : n_parts_{n_parts}, cell_parts_{cell_parts}, node_parts_{node_parts} {
  }

  void Shuffle(CgnsFile* mesh, MapperType* mapper) {
    auto& m_to_c_nodes = mapper->metis_to_cgns_for_nodes;
    auto& m_to_c_cells = mapper->metis_to_cgns_for_cells;
    auto& c_to_m_nodes = mapper->cgns_to_metis_for_nodes;
    auto& c_to_m_cells = mapper->cgns_to_metis_for_cells;
    auto& base = mesh->GetBase(1);
    int n_zones = base.CountZones();
    for (int zid = 1; zid <= n_zones; ++zid) {
      auto& zone = base.GetZone(zid);
      // shuffle nodes and data on nodes
      auto metis_nid_offset = c_to_m_nodes[zid][1];
      auto new_to_old_for_nodes = GetNewOrder(
          &(node_parts_[metis_nid_offset]), zone.CountNodes());
      auto old_to_new_for_nodes = new_to_old_for_nodes;
      for (int i = 0; i < old_to_new_for_nodes.size(); ++i) {
        old_to_new_for_nodes.at(new_to_old_for_nodes.at(i)) = i;
      }
      auto& coord = zone.GetCoordinates();
      ShuffleData(new_to_old_for_nodes, coord.x().data());
      ShuffleData(new_to_old_for_nodes, coord.y().data());
      ShuffleData(new_to_old_for_nodes, coord.z().data());
      ShuffleData(new_to_old_for_nodes, &(c_to_m_nodes[zid][1]));
      ShuffleData(old_to_new_for_nodes, &(m_to_c_nodes[metis_nid_offset]));
      int n_solutions = zone.CountSolutions();
      for (int solution_id = 1; solution_id <= n_solutions; solution_id++) {
        auto& solution = zone.GetSolution(solution_id);
        if (!solution.OnNodes())
          continue;
        for (int i = 1; i <= solution.CountFields(); ++i) {
          auto& field = solution.GetField(i);
          ShuffleData(new_to_old_for_nodes, field.data());
        }
      }
      // shuffle cells and data on cells
      int n_sections = zone.CountSections();
      for (int sid = 1; sid <= n_sections; ++sid) {
        auto& section = zone.GetSection(sid);
        if (section.dim() != base.GetCellDim())
          continue;
        int n_cells = section.CountCells();
        auto range_min = section.CellIdMin();
        auto metis_cid_offset = c_to_m_cells[zid][sid].at(range_min);
        /* Shuffle Connectivity */
        auto new_to_old_for_cells = GetNewOrder(
            &(cell_parts_[metis_cid_offset]), n_cells);
        auto old_to_new_for_cells = new_to_old_for_cells;
        for (int i = 0; i < old_to_new_for_cells.size(); ++i) {
          old_to_new_for_cells.at(new_to_old_for_cells.at(i)) = i;
        }
        ShuffleData(old_to_new_for_cells, &(m_to_c_cells[metis_cid_offset]));
        ShuffleData(new_to_old_for_cells, c_to_m_cells[zid][sid].data());
        int npe = section.CountNodesByType();
        auto* node_id_list = section.GetNodeIdList();
        ShuffleConnectivity(new_to_old_for_nodes, new_to_old_for_cells, npe, node_id_list);
        /* Shuffle Data on Cells */
        int n_solutions = zone.CountSolutions();
        for (int solution_id = 1; solution_id <= n_solutions; solution_id++) {
          auto& solution = zone.GetSolution(solution_id);
          if (!solution.OnCells())
            continue;
          for (int i = 1; i <= solution.CountFields(); ++i) {
            auto& field = solution.GetField(i);
            auto* field_ptr = &(field.at(range_min));
            ShuffleData(new_to_old_for_cells, field_ptr);
          }
        }
      }
    }
  }

 private:
  std::vector<T> const& cell_parts_;
  std::vector<T> const& node_parts_;
  int n_parts_;
};

}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_CGNS_SHUFFLER_HPP_
