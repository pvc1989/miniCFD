// Copyright 2020 Weicheng Pei and Minghao Yang

#ifndef MINI_MESH_CGNS_SHUFFLER_HPP_
#define MINI_MESH_CGNS_SHUFFLER_HPP_

#include <algorithm>
#include <cassert>
#include <cstring>
#include <utility>
#include <vector>

#include "mini/mesh/metis/format.hpp"
#include "mini/mesh/mapper/cgns_to_metis.hpp"

namespace mini {
namespace mesh {

/**
 * @brief Get the New Order object
 * 
 * @tparam T 
 * @param parts 
 * @param n 
 * @return a pair of std::vector<int>
 */
template <typename T = int>
std::pair<std::vector<int>, std::vector<int>>
GetNewOrder(const T* parts, int n) {
  auto new_to_old = std::vector<int>(n);
  auto old_to_new = std::vector<int>(n);
  std::iota(new_to_old.begin(), new_to_old.end(), 0);
  auto cmp = [parts](int lid, int rid) {
    return parts[lid] < parts[rid] || (parts[lid] == parts[rid] && lid < rid);
  };
  std::sort(new_to_old.begin(), new_to_old.end(), cmp);
  for (int i = 0; i < n; ++i)
    old_to_new[new_to_old[i]] = i;
  return {new_to_old, old_to_new};
}
/**
 * @brief 
 * 
 * @tparam T 
 * @param new_to_old 
 * @param old_data 
 */
template <typename T>
void ShuffleData(const std::vector<int>& new_to_old, T* old_data) {
  int n = new_to_old.size();
  std::vector<T> new_data(n);
  for (int i = 0; i < n; ++i)
    new_data[i] = old_data[new_to_old[i]];
  std::memcpy(old_data, new_data.data(), n * sizeof(T));
}
/**
 * @brief 
 * 
 * @tparam T 
 * @param old_to_new_for_nodes 
 * @param new_to_old_for_cells 
 * @param npe 
 * @param old_cid_old_nid 
 */
template <typename T>
void ShuffleConnectivity(const std::vector<int>& old_to_new_for_nodes,
                         const std::vector<int>& new_to_old_for_cells,
                         int npe, T* old_cid_old_nid) {
  int n_cells = new_to_old_for_cells.size();
  int n_nodes = old_to_new_for_nodes.size();
  int node_id_list_size = n_cells * npe;
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

/**
 * @brief 
 * 
 * @tparam Int
 * @tparam Real 
 */
template <typename Int, class Real>
class Shuffler {
 public:
  using CgnsMesh = mini::mesh::cgns::File<Real>;
  using MetisMesh = metis::Mesh<Int>;
  using MapperType = mini::mesh::mapper::CgnsToMetis<Real, Int>;
  using SectionType = mini::mesh::cgns::Section<Real>;
  using SolutionType = mini::mesh::cgns::Solution<Real>;
  using FieldType = mini::mesh::cgns::Field<Real>;

  Shuffler(Int n_parts, std::vector<Int> const& cell_parts,
           std::vector<Int> const& node_parts)
      : n_parts_{n_parts}, cell_parts_{cell_parts}, node_parts_{node_parts} {
  }

  void Shuffle(CgnsMesh* mesh, MapperType* mapper);

 private:
  std::vector<Int> const& cell_parts_;
  std::vector<Int> const& node_parts_;
  Int n_parts_;
};

template <typename Int, class Real>
void Shuffler<Int, Real>::Shuffle(CgnsMesh* mesh, MapperType* mapper) {
  auto& m_to_c_nodes = mapper->metis_to_cgns_for_nodes;
  auto& m_to_c_cells = mapper->metis_to_cgns_for_cells;
  auto& c_to_m_nodes = mapper->cgns_to_metis_for_nodes;
  auto& c_to_m_cells = mapper->cgns_to_metis_for_cells;
  auto& base = mesh->GetBase(1);
  auto n_zones = base.CountZones();
  for (int zid = 1; zid <= n_zones; ++zid) {
    auto& zone = base.GetZone(zid);
    // shuffle nodes and data on nodes
    auto metis_nid_offset = c_to_m_nodes[zid][1];
    auto [new_to_old_for_nodes, old_to_new_for_nodes] = GetNewOrder(
        &(node_parts_[metis_nid_offset]), zone.CountNodes());
    auto& coord = zone.GetCoordinates();
    ShuffleData(new_to_old_for_nodes, coord.x().data());
    ShuffleData(new_to_old_for_nodes, coord.y().data());
    ShuffleData(new_to_old_for_nodes, coord.z().data());
    ShuffleData(new_to_old_for_nodes, &(c_to_m_nodes[zid][1]));
    ShuffleData(old_to_new_for_nodes, &(m_to_c_nodes[metis_nid_offset]));
    auto n_solutions = zone.CountSolutions();
    for (auto solution_id = 1; solution_id <= n_solutions; solution_id++) {
      auto& solution = zone.GetSolution(solution_id);
      if (!solution.OnNodes())
        continue;
      for (auto i = 1; i <= solution.CountFields(); ++i) {
        auto& field = solution.GetField(i);
        ShuffleData(new_to_old_for_nodes, field.data());
      }
    }
    // shuffle cells and data on cells
    auto n_sections = zone.CountSections();
    for (auto sid = 1; sid <= n_sections; ++sid) {
      auto& section = zone.GetSection(sid);
      if (section.dim() != base.GetCellDim())
        continue;
      auto n_cells = section.CountCells();
      auto range_min = section.CellIdMin();
      auto metis_cid_offset = c_to_m_cells[zid][sid].at(range_min);
      /* Shuffle Connectivity */
      auto [new_to_old_for_cells, old_to_new_for_cells] = GetNewOrder(
          &(cell_parts_[metis_cid_offset]), n_cells);
      ShuffleData(old_to_new_for_cells, &(m_to_c_cells[metis_cid_offset]));
      ShuffleData(new_to_old_for_cells, c_to_m_cells[zid][sid].data());
      auto npe = section.CountNodesByType();
      auto* node_id_list = section.GetNodeIdList();
      ShuffleConnectivity(new_to_old_for_nodes, new_to_old_for_cells,
          npe, node_id_list);
      /* Shuffle Data on Cells */
      auto n_solutions = zone.CountSolutions();
      for (auto solution_id = 1; solution_id <= n_solutions; solution_id++) {
        auto& solution = zone.GetSolution(solution_id);
        if (!solution.OnCells())
          continue;
        for (auto i = 1; i <= solution.CountFields(); ++i) {
          auto& field = solution.GetField(i);
          auto* field_ptr = &(field.at(range_min));
          ShuffleData(new_to_old_for_cells, field_ptr);
        }
      }
    }
  }
}

}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_CGNS_SHUFFLER_HPP_
