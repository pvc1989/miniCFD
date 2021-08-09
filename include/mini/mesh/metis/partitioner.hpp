// Copyright 2021 PEI Weicheng and YANG Minghao and JIANG Yuyan

#ifndef MINI_MESH_METIS_PARTITIONER_HPP_
#define MINI_MESH_METIS_PARTITIONER_HPP_

#include <memory>
#include <type_traits>
#include <vector>

#include "metis.h"
#include "mini/mesh/metis/format.hpp"

namespace mini {
namespace mesh {
namespace metis {

/**
 * @brief A wrapper of `METIS_PartGraphKway()`, which partitions a graph into K parts.
 * 
 * @tparam Int index type
 * @tparam Graph sparse graph type
 * @tparam Real real number type
 * 
 * @param[in] n_vertices the number of vertices in the graph
 * @param[in] n_constraints the number of balancing constraints (>= 1)
 * @param[in] graph the graph to be partitioned
 * @param[in] cost_of_each_vertex the computational cost of each vertex
 * @param[in] size_of_each_vertex the communication size of each vertex
 * @param[in] cost_of_each_edge the weight of each edge
 * @param[in] n_parts the number of parts to be partitioned
 * @param[in] weight_of_each_part the weight of each part (sum must be 1.0)
 * @param[in] unbalances the unbalance tolerance for each constraint
 * @param[in] options the array of METIS options
 * @param[out] objective_value the edge cut or the communication volume of the partitioning
 * @param[out] vertex_parts the part id of each vertex
 */
template <typename Int, typename Graph, typename Real>
void PartGraphKway(
    const Int &n_vertices, const Int &n_constraints,
    const Graph &graph,
    const std::vector<Int> &cost_of_each_vertex,
    const std::vector<Int> &size_of_each_vertex,
    const std::vector<Int> &cost_of_each_edge,
    const Int &n_parts,
    const std::vector<Real> &weight_of_each_part,
    const std::vector<Real> &unbalances,
    const std::vector<Int> &options,
    /* input ↑, output ↓ */
    Int *objective_value, std::vector<Int> *vertex_parts) {
  static_assert(std::is_integral_v<Int>, "`Int` must be an integral type.");
  static_assert(std::is_same_v<Int, idx_t>, "`Int` must be `idx_t`.");
  static_assert(std::is_floating_point_v<Real>,
      "`Real` must be a floating-point type.");
  static_assert(std::is_same_v<Real, real_t>, "`Real` must be `real_t`.");
  assert(vertex_parts->size() == n_vertices);
  auto error_code = METIS_PartGraphKway(
      const_cast<Int*>(&n_vertices), const_cast<Int*>(&n_constraints),
      const_cast<Int*>(&(graph.range()[0])),
      const_cast<Int*>(&(graph.index()[0])),
      const_cast<Int*>(cost_of_each_vertex.data()),
      const_cast<Int*>(size_of_each_vertex.data()),
      const_cast<Int*>(cost_of_each_edge.data()),
      const_cast<Int*>(&n_parts),
      const_cast<Real*>(weight_of_each_part.data()),
      const_cast<Real*>(unbalances.data()),
      const_cast<Int*>(options.data()),
      objective_value, vertex_parts->data());
  assert(error_code == METIS_OK);
}
/**
 * @brief A wrapper of `METIS_PartMeshDual()`, which partitions a mesh into K parts.
 * 
 * @tparam Int index type
 * @tparam Real real number type
 * 
 * @param[in] n_cells the number of cells in the mesh
 * @param[in] n_nodes the number of nodes in the mesh
 * @param[in] mesh the mesh to be partitioned
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
void PartMesh(const Mesh<Int> &mesh,
    const std::vector<Int> &cost_of_each_cell,
    const std::vector<Int> &size_of_each_cell,
    Int n_common_nodes, Int n_parts,
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
  Int n_cells = mesh.CountCells();
  Int n_nodes = mesh.CountNodes();
  auto valid = [](auto const& v, Int n){
    return v.size() == 0 || v.size() == n;
  };
  assert(valid(cost_of_each_cell, n_cells));
  assert(valid(size_of_each_cell, n_cells));
  assert(valid(weight_of_each_part, n_parts));
  cell_parts->resize(n_cells);
  node_parts->resize(n_nodes);
  auto error_code = METIS_PartMeshDual(
      &n_cells, &n_nodes,
      const_cast<Int*>(&(mesh.range()[0])),
      const_cast<Int*>(&(mesh.nodes()[0])),
      const_cast<Int*>(cost_of_each_cell.data()),
      const_cast<Int*>(size_of_each_cell.data()),
      const_cast<Int*>(&n_common_nodes),
      &n_parts, const_cast<Real*>(weight_of_each_part.data()),
      const_cast<Int*>(options.data()),
      objective_value, cell_parts->data(), node_parts->data());
  assert(error_code == METIS_OK);
}
/**
 * @brief A wrapper of `METIS_MeshToDual()`, which converts a mesh to its dual graph.
 * 
 * @tparam Int index type
 * 
 * @param[in] n_cells the number of cells in the mesh
 * @param[in] n_nodes the number of nodes in the mesh
 * @param[in] mesh the mesh to be converted
 * @param[in] n_common_nodes the minimum number of nodes shared by two neighboring cells
 * @param[in] index_base the the base of indexing (0 or 1)
 */
template <typename Int>
GraphWithDeleter<Int> MeshToDual(
    const Int &n_cells, const Int &n_nodes,
    const Mesh<Int> &mesh,
    const Int &n_common_nodes, const Int &index_base) {
  Int *range, *neighbors;
  auto error_code = METIS_MeshToDual(
      const_cast<Int*>(&n_cells), const_cast<Int*>(&n_nodes),
      const_cast<Int*>(&(mesh.range()[0])),
      const_cast<Int*>(&(mesh.nodes()[0])),
      const_cast<Int*>(&n_common_nodes), const_cast<Int*>(&index_base),
      &range, &neighbors);
  assert(error_code == METIS_OK);
  return GraphWithDeleter<Int>(n_cells, range, neighbors);
}

}  // namespace metis
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_METIS_PARTITIONER_HPP_
