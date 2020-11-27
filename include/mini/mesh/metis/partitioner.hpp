// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_MESH_METIS_PARTITIONER_HPP_
#define MINI_MESH_METIS_PARTITIONER_HPP_

#include <memory>
#include <type_traits>
#include <vector>

#include "metis.h"

namespace mini {
namespace mesh {
namespace metis {

template <typename Int, typename IntArray, typename Real>
int PartGraphKway(
    const Int &n_nodes,
    const Int &n_constraints,
    const IntArray &range_of_each_node,
    const IntArray &neighbors_of_each_node,
    const std::vector<Int> &cost_of_each_node,  /* computational cost */
    const std::vector<Int> &size_of_each_node,  /* communication size */
    const std::vector<Int> &cost_of_each_edge,  /* weight of each edge */
    const Int &n_parts,
    const std::vector<Real> &weight_of_each_part,  /* sum must be 1.0 */
    const std::vector<Real> &unbalances,  /* unbalance tolerance */
    const std::vector<Int> &options,
    // output:
    Int *objective_value,  /* edge cut or communication volume */
    std::vector<Int> *node_parts) {
  static_assert(std::is_integral_v<Int>, "`Int` must be an integral type.");
  static_assert(std::is_same_v<Int, idx_t>, "`Int` must be `idx_t`.");
  static_assert(std::is_floating_point_v<Real>, "`Real` must be a floating-point type.");
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
      objective_value, node_parts->data()
  );
}
template <typename Int, typename Real>
int PartMeshDual(
    const Int &n_cells,
    const Int &n_nodes,
    const std::vector<Int> &range_of_each_cell,
    const std::vector<Int> &nodes_in_each_cell,
    const std::vector<Int> &cost_of_each_cell,  /* computational cost */
    const std::vector<Int> &size_of_each_cell,  /* communication size */
    const Int &n_common_nodes,
    const Int &n_parts,
    const std::vector<Real> &weight_of_each_part,  /* sum must be 1.0 */
    const std::vector<Int> &options,
    // output:
    Int *objective_value,  /* edge cut or communication volume */
    std::vector<Int> *cell_parts,
    std::vector<Int> *node_parts) {
  static_assert(std::is_integral_v<Int>, "`Int` must be an integral type.");
  static_assert(std::is_same_v<Int, idx_t>, "`Int` must be `idx_t`.");
  static_assert(std::is_floating_point_v<Real>, "`Real` must be a floating-point type.");
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
      objective_value, cell_parts->data(), node_parts->data()
  );
}
template <typename Int>
int MeshToDual(
    const Int &n_cells,
    const Int &n_nodes,
    const std::vector<Int> &range_of_each_cell,
    const std::vector<Int> &nodes_in_each_cell,
    const Int &n_common_nodes,
    const Int &index_base,  /* 0 or 1 */
    // output:
    std::unique_ptr<Int[]> *range_of_each_dual_vertex,
    std::unique_ptr<Int[]> *neighbors_of_each_dual_vertex) {
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
