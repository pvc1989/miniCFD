// Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_MESH_METIS_PARTITIONER_HPP_
#define MINI_MESH_METIS_PARTITIONER_HPP_

#include <type_traits>
#include <vector>

#include "metis.h"

namespace mini {
namespace mesh {
namespace metis {

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

}  // namespace metis
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_METIS_PARTITIONER_HPP_
