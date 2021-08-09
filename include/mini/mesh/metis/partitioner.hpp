// Copyright 2021 PEI Weicheng and YANG Minghao and JIANG Yuyan

#ifndef MINI_MESH_METIS_PARTITIONER_HPP_
#define MINI_MESH_METIS_PARTITIONER_HPP_

#include <memory>
#include <type_traits>
#include <vector>

#include "metis.h"

namespace mini {
namespace mesh {
namespace metis {

/**
 * @brief A wrapper of `METIS_PartGraphKway()`, which partitions a graph into k parts.
 * 
 * @tparam Int index type
 * @tparam IntArray index array type
 * @tparam Real real number type
 * 
 * @param[in] n_nodes the number of nodes in the graph
 * @param[in] n_constraints the number of balancing constraints (>= 1)
 * @param[in] range_of_each_node the range of indices of each node's list of neighbors
 * @param[in] neighbors_of_each_node the list of neighbors of each node
 * @param[in] cost_of_each_node the computational cost of each node
 * @param[in] size_of_each_node the communication size of each node
 * @param[in] cost_of_each_edge the weight of each edge
 * @param[in] n_parts the number of parts to be partitioned
 * @param[in] weight_of_each_part the weight of each part (sum must be 1.0)
 * @param[in] unbalances the unbalance tolerance for each constraint
 * @param[in] options the array of METIS options
 * @param[out] objective_value the edge cut or the communication volume of the partitioning
 * @param[out] node_parts the part id of each node
 */
template <typename Int, typename IntArray, typename Real>
int PartGraphKway(
    const Int &n_nodes,
    const Int &n_constraints,
    const IntArray &range_of_each_node,
    const IntArray &neighbors_of_each_node,
    const std::vector<Int> &cost_of_each_node,
    const std::vector<Int> &size_of_each_node,
    const std::vector<Int> &cost_of_each_edge,
    const Int &n_parts,
    const std::vector<Real> &weight_of_each_part,
    const std::vector<Real> &unbalances,
    const std::vector<Int> &options,
    /* input ↑, output ↓ */
    Int *objective_value,
    std::vector<Int> *node_parts) {
  static_assert(std::is_integral_v<Int>, "`Int` must be an integral type.");
  static_assert(std::is_same_v<Int, idx_t>, "`Int` must be `idx_t`.");
  static_assert(std::is_floating_point_v<Real>,
      "`Real` must be a floating-point type.");
  static_assert(std::is_same_v<Real, real_t>, "`Real` must be `real_t`.");
  assert(node_parts->size() == n_nodes);
  return METIS_PartGraphKway(
      const_cast<Int*>(&n_nodes), const_cast<Int*>(&n_constraints),
      const_cast<Int*>(&range_of_each_node[0]),
      const_cast<Int*>(&neighbors_of_each_node[0]),
      const_cast<Int*>(cost_of_each_node.data()),
      const_cast<Int*>(size_of_each_node.data()),
      const_cast<Int*>(cost_of_each_edge.data()),
      const_cast<Int*>(&n_parts),
      const_cast<Real*>(weight_of_each_part.data()),
      const_cast<Real*>(unbalances.data()),
      const_cast<Int*>(options.data()),
      objective_value, node_parts->data());
}
/**
 * @brief A wrapper of `METIS_PartMeshDual()`, which partitions a mesh into k parts.
 * 
 * @tparam Int index type
 * @tparam Real real number type
 * 
 * @param[in] n_cells the number of cells in the mesh
 * @param[in] n_nodes the number of nodes in the mesh
 * @param[in] range_of_each_cell the range of indices of each cell's node id list
 * @param[in] nodes_of_each_cell the list of nodes of each cell
 * @param[in] cost_of_each_cell the computational cost of each cell
 * @param[in] size_of_each_cell the communication size of each cell
 * @param[in] n_common_nodes the minimum number of nodes shared by two neighboring cells
 * @param[in] n_parts the number of parts to be partitioned
 * @param[in] weight_of_each_part the weight of each part (sum must be 1.0)
 * @param[in] options the array of METIS options
 * @param[out] objective_value the edge cut or the communication volume of the partitioning
 * @param[out] cell_parts the part id of each cell
 * @param[out] node_parts the part id of each node
 */
template <typename Int, typename Real>
int PartMeshDual(
    const Int &n_cells, const Int &n_nodes,
    const std::vector<Int> &range_of_each_cell,
    const std::vector<Int> &nodes_in_each_cell,
    const std::vector<Int> &cost_of_each_cell,
    const std::vector<Int> &size_of_each_cell,
    const Int &n_common_nodes, const Int &n_parts,
    const std::vector<Real> &weight_of_each_part,
    const std::vector<Int> &options,
    /* input ↑, output ↓ */
    Int *objective_value,
    std::vector<Int> *cell_parts, std::vector<Int> *node_parts) {
  static_assert(std::is_integral_v<Int>, "`Int` must be an integral type.");
  static_assert(std::is_same_v<Int, idx_t>, "`Int` must be `idx_t`.");
  static_assert(std::is_floating_point_v<Real>,
      "`Real` must be a floating-point type.");
  static_assert(std::is_same_v<Real, real_t>, "`Real` must be `real_t`.");
  assert(cell_parts->size() == n_cells);
  assert(node_parts->size() == n_nodes);
  return METIS_PartMeshDual(
      const_cast<Int*>(&n_cells), const_cast<Int*>(&n_nodes),
      const_cast<Int*>(range_of_each_cell.data()),
      const_cast<Int*>(nodes_in_each_cell.data()),
      const_cast<Int*>(cost_of_each_cell.data()),
      const_cast<Int*>(size_of_each_cell.data()),
      const_cast<Int*>(&n_common_nodes),
      const_cast<Int*>(&n_parts),
      const_cast<Real*>(weight_of_each_part.data()),
      const_cast<Int*>(options.data()),
      objective_value, cell_parts->data(), node_parts->data());
}
/**
 * @brief A wrapper of `METIS_Free()`.
 * 
 */
auto deleter = [](void* p){ METIS_Free(p); };
using Deleter = decltype(deleter);
/**
 * @brief A wrapper of `METIS_MeshToDual()`, which converts a mesh to its dual graph.
 * 
 * @tparam Int index type
 * 
 * @param[in] n_cells the number of cells in the mesh
 * @param[in] n_nodes the number of nodes in the mesh
 * @param[in] range_of_each_cell the range of indices of each cell's node id list
 * @param[in] nodes_of_each_cell the list of nodes of each cell
 * @param[in] n_common_nodes the minimum number of nodes shared by two neighboring cells
 * @param[in] index_base the the base of indexing (0 or 1)
 * @param[out] range_of_each_dual_vertex the range of indices of each cell's list of neighbors
 * @param[out] neighbors_of_each_dual_vertex the list of neighbors of each cell
 */
template <typename Int>
int MeshToDual(
    const Int &n_cells, const Int &n_nodes,
    const std::vector<Int> &range_of_each_cell,
    const std::vector<Int> &nodes_in_each_cell,
    const Int &n_common_nodes, const Int &index_base,
    /* input ↑, output ↓ */
    std::unique_ptr<Int[], Deleter> *range_of_each_dual_vertex,
    std::unique_ptr<Int[], Deleter> *neighbors_of_each_dual_vertex) {
  Int *range, *neighbors;
  auto error_code = METIS_MeshToDual(
      const_cast<Int*>(&n_cells), const_cast<Int*>(&n_nodes),
      const_cast<Int*>(range_of_each_cell.data()),
      const_cast<Int*>(nodes_in_each_cell.data()),
      const_cast<Int*>(&n_common_nodes), const_cast<Int*>(&index_base),
      &range, &neighbors);
  range_of_each_dual_vertex->reset(range);
  neighbors_of_each_dual_vertex->reset(neighbors);
  return error_code;
}

}  // namespace metis
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_METIS_PARTITIONER_HPP_
