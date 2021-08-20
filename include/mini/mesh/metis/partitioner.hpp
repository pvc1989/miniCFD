// Copyright 2021 PEI Weicheng and YANG Minghao and JIANG Yuyan

#ifndef MINI_MESH_METIS_PARTITIONER_HPP_
#define MINI_MESH_METIS_PARTITIONER_HPP_

#include <memory>
#include <type_traits>
#include <vector>
#include <utility>

#include "metis.h"
#include "mini/mesh/metis/format.hpp"

namespace mini {
namespace mesh {
namespace metis {

template <class Container>
static inline bool valid(Container&& c, std::size_t size) {
  return c.size() == 0 || c.size() == size;
}

/**
 * @brief A wrapper of `METIS_PartGraphKway()`, which partitions a graph into K parts.
 * 
 * @tparam Int index type
 * @tparam Graph sparse graph type
 * @tparam Real real number type
 * 
 * @param[in] graph the graph to be partitioned
 * @param[in] n_parts the number of parts to be partitioned
 * @param[in] n_constraints the number of balancing constraints (>= 1)
 * @param[in] cost_of_each_vertex the computational cost of each vertex
 * @param[in] size_of_each_vertex the communication size of each vertex
 * @param[in] cost_of_each_edge the weight of each edge
 * @param[in] weight_of_each_part the weight of each part (sum must be 1.0)
 * @param[in] unbalances the unbalance tolerance for each constraint
 * @param[in] options the array of METIS options
 * @return the part id of each vertex
 */
template <typename Graph, typename Int>
std::vector<Int> PartGraph(
    const Graph &graph, Int n_parts, Int n_constraints = 1,
    const std::vector<Int> &cost_of_each_vertex = {},
    const std::vector<Int> &size_of_each_vertex = {},
    const std::vector<Int> &cost_of_each_edge = {},
    const std::vector<real_t> &weight_of_each_part = {},
    const std::vector<real_t> &unbalances = {},
    const std::vector<Int> &options = {}) {
  static_assert(sizeof(Int) == sizeof(idx_t), "`Int` and `idx_t` must have the same size.");
  Int n_vertices = graph.CountVertices();
  auto vertex_parts = std::vector<Int>(n_vertices);
  assert(valid(cost_of_each_vertex, n_vertices));
  assert(valid(size_of_each_vertex, n_vertices));
  assert(valid(cost_of_each_edge, graph.CountEdges()));
  assert(valid(weight_of_each_part, n_parts));
  assert(valid(unbalances, n_parts));
  Int objective_value;
  auto error_code = METIS_PartGraphKway(
      &n_vertices, &n_constraints,
      const_cast<Int*>(&(graph.range(0))),
      const_cast<Int*>(&(graph.index(0))),
      const_cast<Int*>(cost_of_each_vertex.data()),
      const_cast<Int*>(size_of_each_vertex.data()),
      const_cast<Int*>(cost_of_each_edge.data()),
      const_cast<Int*>(&n_parts),
      const_cast<real_t*>(weight_of_each_part.data()),
      const_cast<real_t*>(unbalances.data()),
      const_cast<Int*>(options.data()),
      &objective_value, vertex_parts.data());
  assert(error_code == METIS_OK);
  return vertex_parts;
}
/**
 * @brief A wrapper of `METIS_PartMeshDual()`, which partitions a mesh into K parts.
 * 
 * @tparam Int index type
 * @tparam Real real number type
 * 
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
template <typename Int>
std::pair<std::vector<Int>, std::vector<Int>> PartMesh(
    const Mesh<Int> &mesh, Int n_parts, Int n_common_nodes = 2,
    const std::vector<Int> &cost_of_each_cell = {},
    const std::vector<Int> &size_of_each_cell = {},
    const std::vector<real_t> &weight_of_each_part = {},
    const std::vector<Int> &options = {}) {
  static_assert(sizeof(Int) == sizeof(idx_t),
      "`Int` and `idx_t` must have the same size.");
  Int n_cells = mesh.CountCells();
  Int n_nodes = mesh.CountNodes();
  assert(valid(cost_of_each_cell, n_cells));
  assert(valid(size_of_each_cell, n_cells));
  assert(valid(weight_of_each_part, n_parts));
  std::vector<Int> cell_parts(n_cells), node_parts(n_nodes);
  Int objective_value;
  auto error_code = METIS_PartMeshDual(
      &n_cells, &n_nodes,
      const_cast<Int*>(&(mesh.range(0))),
      const_cast<Int*>(&(mesh.nodes(0))),
      const_cast<Int*>(cost_of_each_cell.data()),
      const_cast<Int*>(size_of_each_cell.data()),
      &n_common_nodes, &n_parts,
      const_cast<real_t*>(weight_of_each_part.data()),
      const_cast<Int*>(options.data()),
      &objective_value, cell_parts.data(), node_parts.data());
  assert(error_code == METIS_OK);
  return {cell_parts, node_parts};
}
/**
 * @brief A wrapper of `METIS_MeshToDual()`, which converts a mesh to its dual graph.
 * 
 * @tparam Int index type
 * 
 * @param[in] mesh the mesh to be converted
 * @param[in] n_common_nodes the minimum number of nodes shared by two neighboring cells
 * @param[in] index_base the the base of indexing (0 or 1)
 */
template <typename Int>
SparseGraphWithDeleter<Int> MeshToDual(const Mesh<Int> &mesh,
    Int n_common_nodes, Int index_base = 0) {
  Int n_cells = mesh.CountCells();
  Int n_nodes = mesh.CountNodes();
  Int *range, *neighbors;
  auto error_code = METIS_MeshToDual(
      &n_cells, &n_nodes,
      const_cast<Int*>(&(mesh.range(0))),
      const_cast<Int*>(&(mesh.nodes(0))),
      &n_common_nodes, &index_base,
      &range, &neighbors);
  assert(error_code == METIS_OK);
  return SparseGraphWithDeleter<Int>(n_cells, range, neighbors);
}
/**
 * @brief Get the partition of nodes from the partition of cells.
 * 
 * @tparam Int 
 * @param mesh 
 * @param cell_parts 
 * @param n_parts 
 * @return std::vector<Int> 
 */
template <typename Int>
std::vector<Int> GetNodeParts(
    const metis::Mesh<Int>& mesh, const std::vector<Int>& cell_parts,
    Int n_parts) {
  Int n_nodes = mesh.CountNodes();
  Int n_cells = mesh.CountCells();
  auto node_parts = std::vector<Int>(n_nodes, n_parts);
  auto node_pointer = &(mesh.nodes(0));
  auto curr_range_pointer = &(mesh.range(0));
  for (Int i = 0; i < n_cells; ++i) {
    auto part_value = cell_parts[i];
    auto head = *curr_range_pointer++;
    auto tail = *curr_range_pointer;
    for (Int j = head; j < tail; ++j) {
      Int node_index = *node_pointer++;
      /* If more than one parts share a common node, then the
      node belongs to the min-id part. */
      if (part_value < node_parts[node_index])
        node_parts[node_index] = part_value;
    }
  }
  return node_parts;
}

}  // namespace metis
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_METIS_PARTITIONER_HPP_
