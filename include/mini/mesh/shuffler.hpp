// Copyright 2020 Weicheng Pei and Minghao Yang

#ifndef MINI_MESH_SHUFFLER_HPP_
#define MINI_MESH_SHUFFLER_HPP_

#include <algorithm>
#include <cassert>
#include <cstring>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mini/mesh/cgns.hpp"
#include "mini/mesh/metis.hpp"
#include "mini/mesh/mapper.hpp"

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
// template <typename T>
// void UpdateNodeIdList(const std::vector<int>& old_to_new_for_nodes,
//                       int list_length, T* node_id_list) {
//   auto* tail = node_id_list + list_length;
//   auto* curr = node_id_list;
//   while (curr != tail) {
//     auto i_node_old = *curr;
//     auto i_node_new = old_to_new_for_nodes[i_node_old];
//     *curr++ = i_node_new;
//   }
// }
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
  int i_node_list_size = n_cells * npe;
  auto old_cid_new_nid = std::vector<T>(i_node_list_size);
  for (int i = 0; i < i_node_list_size; ++i) {
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
  std::map<Int, std::map<Int, cgns::ShiftedVector<Int>>> face_parts_;
  Int n_parts_;

  void FillFaceParts(const CgnsMesh& mesh, const MapperType& mapper);
};


template <typename Int, class Real>
void Shuffler<Int, Real>::FillFaceParts(
    const CgnsMesh& mesh, const MapperType& mapper) {
  auto& m_to_c_nodes = mapper.metis_to_cgns_for_nodes;
  auto& m_to_c_cells = mapper.metis_to_cgns_for_cells;
  auto& c_to_m_nodes = mapper.cgns_to_metis_for_nodes;
  auto& c_to_m_cells = mapper.cgns_to_metis_for_cells;
  auto& base = mesh.GetBase(1);
  auto n_zones = base.CountZones();
  // For each node, find its users' part and append to the node's vector:
  auto node_user_parts = std::vector<std::vector<Int>>(node_parts_.size());
  for (int i_zone = 1; i_zone <= n_zones; ++i_zone) {
    auto& zone = base.GetZone(i_zone);
    auto& i_node_to_m_node = c_to_m_nodes.at(i_zone);
    auto n_sects = zone.CountSections();
    for (int i_sect = 1; i_sect <= n_sects; ++i_sect) {
      auto& sect = zone.GetSection(i_sect);
      if (sect.dim() == base.GetCellDim()) {
        auto npe = sect.CountNodesByType();
        auto n_cells = sect.CountCells();
        auto i_cell_min = sect.CellIdMin();
        auto i_cell_max = sect.CellIdMax();
        auto& i_cell_to_m_cell = c_to_m_cells.at(i_zone).at(i_sect);
        auto* head = sect.GetNodeIdList();
        auto* curr = head;
        auto* tail = head + n_cells * npe;
        for (int i_cell = i_cell_min; i_cell <= i_cell_max; ++i_cell) {
          auto m_cell = i_cell_to_m_cell.at(i_cell);
          auto cell_part = cell_parts_.at(m_cell);
          for (int k = 0; k < npe; ++k) {
            assert(curr + k < tail);
            auto m_node = i_node_to_m_node.at(curr[k]);
            node_user_parts[m_node].emplace_back(cell_part);
          }
          curr += npe;
        }
        assert(curr == tail);
      }
    }
  }
  for (auto& user_parts : node_user_parts) {
    std::sort(user_parts.begin(), user_parts.end());
    auto last = std::unique(user_parts.begin(), user_parts.end());
    user_parts.erase(last, user_parts.end());
  }
  // For each face, determine its part by its nodes:
  for (int i_zone = 1; i_zone <= n_zones; ++i_zone) {
    auto& zone = base.GetZone(i_zone);
    auto& i_node_to_m_node = c_to_m_nodes.at(i_zone);
    auto n_sects = zone.CountSections();
    for (int i_sect = 1; i_sect <= n_sects; ++i_sect) {
      auto& sect = zone.GetSection(i_sect);
      if (sect.dim() + 1 == base.GetCellDim()) {
        auto npe = sect.CountNodesByType();
        auto n_faces = sect.CountCells();
        auto i_face_min = sect.CellIdMin();
        auto i_face_max = sect.CellIdMax();
        auto& i_face_to_part = (face_parts_[i_zone][i_sect]
            = cgns::ShiftedVector<Int>(n_faces, i_face_min));
        auto* head = sect.GetNodeIdList();
        auto* curr = head;
        auto* tail = head + n_faces * npe;
        for (int i_face = i_face_min; i_face <= i_face_max; ++i_face) {
          auto face_part_cnts = std::unordered_map<Int, Int>();
          for (int k = 0; k < npe; ++k) {
            assert(curr + k < tail);
            auto i_node = curr[k];
            auto m_node = i_node_to_m_node.at(i_node);
            for (Int part : node_user_parts[m_node]) {
              face_part_cnts[part]++;
            }
          }
          for (auto [part, cnt] : face_part_cnts) {
            assert(cnt <= npe);
            if (cnt == npe) {
              i_face_to_part.at(i_face) = part;
              break;
            }
          }
          curr += npe;
        }
        assert(curr == tail);
      }
    }
  }
}

template <typename Int, class Real>
void Shuffler<Int, Real>::Shuffle(CgnsMesh* mesh, MapperType* mapper) {
  FillFaceParts(*mesh, *mapper);
  auto& m_to_c_nodes = mapper->metis_to_cgns_for_nodes;
  auto& m_to_c_cells = mapper->metis_to_cgns_for_cells;
  auto& c_to_m_nodes = mapper->cgns_to_metis_for_nodes;
  auto& c_to_m_cells = mapper->cgns_to_metis_for_cells;
  auto& base = mesh->GetBase(1);
  auto n_zones = base.CountZones();
  // shuffle nodes, cells and data on them
  for (int i_zone = 1; i_zone <= n_zones; ++i_zone) {
    auto& zone = base.GetZone(i_zone);
    // shuffle node coordinates
    auto metis_nid_offset = c_to_m_nodes[i_zone][1];
    auto [new_to_old_for_nodes, old_to_new_for_nodes] = GetNewOrder(
        &(node_parts_[metis_nid_offset]), zone.CountNodes());
    auto& coord = zone.GetCoordinates();
    ShuffleData(new_to_old_for_nodes, coord.x().data());
    ShuffleData(new_to_old_for_nodes, coord.y().data());
    ShuffleData(new_to_old_for_nodes, coord.z().data());
    ShuffleData(new_to_old_for_nodes, &(c_to_m_nodes[i_zone][1]));
    ShuffleData(old_to_new_for_nodes, &(m_to_c_nodes[metis_nid_offset]));
    // shuffle data on nodes
    auto n_solns = zone.CountSolutions();
    for (auto i_soln = 1; i_soln <= n_solns; i_soln++) {
      auto& solution = zone.GetSolution(i_soln);
      if (!solution.OnNodes())
        continue;
      for (auto i = 1; i <= solution.CountFields(); ++i) {
        auto& field = solution.GetField(i);
        ShuffleData(new_to_old_for_nodes, field.data());
      }
    }
    // shuffle cells and data on cells
    auto n_sects = zone.CountSections();
    for (auto i_sect = 1; i_sect <= n_sects; ++i_sect) {
      auto& section = zone.GetSection(i_sect);
      /* Shuffle Connectivity */
      auto npe = section.CountNodesByType();
      auto n_cells = section.CountCells();
      auto* i_node_list = section.GetNodeIdList();
      if (section.dim() != base.GetCellDim()) {
        // For a lower-dim section, get each face's partition
        auto [new_to_old_for_faces, old_to_new_for_faces] = GetNewOrder(
            face_parts_.at(i_zone).at(i_sect).data(), section.CountCells());
        // then shuffle its connectivity
        ShuffleConnectivity(old_to_new_for_nodes, new_to_old_for_faces,
            npe, i_node_list);
        continue;
      }
      auto range_min = section.CellIdMin();
      auto metis_cid_offset = c_to_m_cells[i_zone][i_sect].at(range_min);
      auto [new_to_old_for_cells, old_to_new_for_cells] = GetNewOrder(
          &(cell_parts_[metis_cid_offset]), n_cells);
      ShuffleData(old_to_new_for_cells, &(m_to_c_cells[metis_cid_offset]));
      ShuffleData(new_to_old_for_cells, c_to_m_cells[i_zone][i_sect].data());
      ShuffleConnectivity(old_to_new_for_nodes, new_to_old_for_cells,
          npe, i_node_list);
      /* Shuffle Data on Cells */
      auto n_solns = zone.CountSolutions();
      for (auto i_soln = 1; i_soln <= n_solns; i_soln++) {
        auto& solution = zone.GetSolution(i_soln);
        if (!solution.OnCells())
          continue;
        for (auto i = 1; i <= solution.CountFields(); ++i) {
          auto& field = solution.GetField(i);
          auto* field_ptr = &(field.at(range_min));
          ShuffleData(new_to_old_for_cells, field_ptr);
        }
      }
    }
    zone.UpdateSectionRanges();
  }
}

}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_SHUFFLER_HPP_
