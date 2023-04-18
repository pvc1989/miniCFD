// Copyright 2021 PEI Weicheng and YANG Minghao and JIANG Yuyan

#ifndef MINI_MESH_METIS_HPP_
#define MINI_MESH_METIS_HPP_

#include <algorithm>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "metis.h"

namespace mini {
namespace mesh {
namespace metis {

/**
 * @brief The data structure for representing sparse graphs.
 * 
 * In a real CSR (Compressed Sparse Row) matrix, the column indices for row `i` are stored in `index_[ range_[i], range_[i+1] )` and their corresponding values are stored in `value_[ range_[i], range_[i+1] )`.
 * To represent an unweighted graph, we only care about the positions of non-zero values, so we have ignored the `value_` array.
 * 
 * @tparam Int index type
 */
template <typename Int>
class SparseMatrix {
 private:
  static_assert(std::is_integral_v<Int>, "Integral required.");
  std::vector<Int> range_, index_/* , value_ */;

 public:
  SparseMatrix() = default;
  SparseMatrix(std::vector<Int> const &range, std::vector<Int> const &index)
      : range_(range), index_(index) {
  }

 public:
  Int const &range(Int i) const {
    return range_[i];
  }
  Int const &index(Int i) const {
    return index_[i];
  }
  Int CountVertices() const {
    return range_.size() - 1;
  }
  Int CountEdges() const {
    return index_.size();
  }
  Int &range(Int i) {
    return range_[i];
  }
  Int &index(Int i) {
    return index_[i];
  }
  /**
   * @brief Resize the underling containers.
   * 
   * @param n_vertices the number of vertices in this graph
   * @param n_edges the number of edges in this graph
   */
  void resize(Int n_vertices, Int n_edges) {
    range_.resize(n_vertices + 1);
    index_.resize(n_edges);
  }
};

/**
 * @brief The data structure for representing meshes.
 * 
 * @tparam Int the index type
 */
template <typename Int>
class Mesh : private SparseMatrix<Int> {
  static_assert(std::is_integral_v<Int>, "Integral required.");
  using Base = SparseMatrix<Int>;
  Int n_nodes_;

 public:
  Mesh() = default;
  Mesh(std::vector<Int> const &range,
       std::vector<Int> const &index, Int n_nodes)
      : Base(range, index), n_nodes_(n_nodes) {
    assert(n_nodes > *(std::max_element(index.begin(), index.end())));
  }

 public:
  Int const &range(Int i) const {
    return this->Base::range(i);
  }
  Int const &nodes(Int i) const {
    return this->index(i);
  }
  Int CountCells() const {
    return this->CountVertices();
  }
  Int CountNodes() const {
    return n_nodes_;
  }
  Int &range(Int i) {
    return this->Base::range(i);
  }
  Int &nodes(Int i) {
    return this->index(i);
  }
  /**
   * @brief Resize the underling containers.
   * 
   * @param n_cells the number of cells in this mesh
   * @param n_nodes_global the number of nodes counted globally (without duplication)
   * @param n_nodes_local the number of nodes counted locally (with duplication)
   */
  void resize(Int n_cells, Int n_nodes_global, Int n_nodes_local) {
    assert(0 < n_cells);
    assert(0 < n_nodes_global);
    assert(0 < n_nodes_local);
    n_nodes_ = n_nodes_global;
    Base::resize(n_cells, n_nodes_local);
  }
};

/**
 * @brief The data structure for representing sparse graphs with deleter.
 * 
 * @tparam Int the index type
 */
template <typename Int>
class SparseGraphWithDeleter {
  static_assert(std::is_integral_v<Int>, "Integral required.");
  Int size_, *range_, *index_/* , *value_ */;

 public:
  /**
   * @brief Construct a new Sparse Graph With Deleter object.
   * 
   * @param size the number of vertices
   * @param range the address of the explicitly allocated range array
   * @param index the address of the explicitly allocated index array
   */
  SparseGraphWithDeleter(Int size, Int *range, Int *index)
      : size_(size), range_(range), index_(index) {
  }
  /**
   * @brief Destroy the Sparse Graph With Deleter object
   *
   * Explicitly free the arrays, which are allocated by METIS. 
   */
  ~SparseGraphWithDeleter() noexcept {
    int errors = (METIS_Free(range_) != METIS_OK)
               + (METIS_Free(index_) != METIS_OK);
    assert(errors == 0);
  }
  Int const &range(Int i) const {
    return range_[i];
  }
  Int const &index(Int i) const {
    return index_[i];
  }
  Int CountVertices() const {
    return size_;
  }
  Int CountEdges() const {
    return range_[size_];
  }
  Int &range(Int i) {
    return range_[i];
  }
  Int &index(Int i) {
    return index_[i];
  }
};

template <class Container>
static inline bool valid(Container &&c, std::size_t size) {
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
  static_assert(sizeof(Int) == sizeof(idx_t),
      "`Int` and `idx_t` must have the same size.");
  Int n_vertices = graph.CountVertices();
  auto vertex_parts = std::vector<Int>(n_vertices);
  if (n_parts == 1)
    return vertex_parts;
  assert(valid(cost_of_each_vertex, n_vertices));
  assert(valid(size_of_each_vertex, n_vertices));
  assert(valid(cost_of_each_edge, graph.CountEdges()));
  assert(valid(weight_of_each_part, n_parts));
  assert(valid(unbalances, n_parts));
  Int objective_value;
  auto error_code = METIS_PartGraphKway(
      &n_vertices, &n_constraints,
      const_cast<Int *>(&(graph.range(0))),
      const_cast<Int *>(&(graph.index(0))),
      const_cast<Int *>(cost_of_each_vertex.data()),
      const_cast<Int *>(size_of_each_vertex.data()),
      const_cast<Int *>(cost_of_each_edge.data()),
      const_cast<Int *>(&n_parts),
      const_cast<real_t *>(weight_of_each_part.data()),
      const_cast<real_t *>(unbalances.data()),
      const_cast<Int *>(options.data()),
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
      const_cast<Int *>(&(mesh.range(0))),
      const_cast<Int *>(&(mesh.nodes(0))),
      const_cast<Int *>(cost_of_each_cell.data()),
      const_cast<Int *>(size_of_each_cell.data()),
      &n_common_nodes, &n_parts,
      const_cast<real_t *>(weight_of_each_part.data()),
      const_cast<Int *>(options.data()),
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
      const_cast<Int *>(&(mesh.range(0))),
      const_cast<Int *>(&(mesh.nodes(0))),
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

#endif  // MINI_MESH_METIS_HPP_
