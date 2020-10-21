// Copyright 2020 Weicheng Pei and Minghao Yang

#ifndef MINI_MESH_METIS_SHUFFLER_HPP_
#define MINI_MESH_METIS_SHUFFLER_HPP_

#include <cassert>
#include <cstdio>
#include <map>
#include <memory>
#include <set>
#include <vector>

#include "cgnslib.h"

namespace mini {
namespace mesh {
namespace metis {

using CSRM = mini::mesh::cgns::CompressedSparseRowMatrix<idx_t>;
template <typename T>
void GetNodePartsByConnectivity(const CSRM& cell_csrm, const std::vector<T>& cell_parts,
                                T n_parts, int n_nodes, std::vector<T>& node_parts) {
  node_parts.resize(n_nodes, n_parts);
  auto node_pointer = cell_csrm.index.data();
  int n_cells = cell_csrm.pointer.size() - 1;
  auto curr_range_pointer = cell_csrm.pointer.data();
  for (int i = 0; i < n_cells; ++i) {
    auto part_value = cell_parts[i];
    auto head = *curr_range_pointer++;
    auto tail = *curr_range_pointer;
    for (int j = head; j < tail; ++j) {
      int node_index = *node_pointer++;
      if (part_value < node_parts[node_index]) node_parts[node_index] = part_value;
    }
  }
}

template <typename T>
void SelectByIndices(const std::vector<T>& global_parts,
                     std::vector<int>& selected_indices,
                     std::vector<T>& local_parts) {
  int n = selected_indices.size();
  local_parts.reserve(n);
  for (int i = 0; i < n; ++i) {
    local_parts[i] = global_parts[selected_indices[i]];
  }                  
}

template <typename T>
void ReorderByParts(const std::vector<T>& parts, std::vector<int>& new_order) {
  std::map<int, std::set<int>> collector;
  int n = parts.size();
  new_order.reserve(n);
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
      new_order[index++] = local_id;
    }
  }
}

template <typename T1, typename T2>
void ShuffleDataArray(const std::vector<T1>& old_array,
                      const std::vector<T2>& new_order,
                      std::vector<T1>& new_array) {
 int n = old_array.size();
 new_array.reserve(n);
 for (int i = 0; i < n; ++i) {
   new_array[i] = old_array[new_order[i]];
 }
}

template <typename T>
void ShuffleConnectivity(const std::vector<T>& old_to_new_for_node,
                         const std::vector<int>& new_cell_order,
                         const std::vector<T>& old_connectivity,
                         std::vector<T>& new_connectivity) {
  int connectivity_size = old_connectivity.size();
  int n_cells = new_cell_order.size();
  new_connectivity.reserve(connectivity_size);
  int n_nodes_per_cell = connectivity_size / n_cells;
  auto curr_ptr = new_connectivity.data();
  for (int i = 0; i < n_cells; ++i) {
    int range_min = n_nodes_per_cell * new_cell_order[i];
    for (int j = range_min; j < range_min + n_nodes_per_cell; ++j) {
      *curr_ptr++ = old_to_new_for_node[old_connectivity[j]];
    }
  }                        
}

}  // namespace metis
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_METIS_SHUFFLER_HPP_
