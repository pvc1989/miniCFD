// Copyright 2021 PEI Weicheng and YANG Minghao and JIANG Yuyan

#ifndef MINI_MESH_METIS_FORMAT_HPP_
#define MINI_MESH_METIS_FORMAT_HPP_

#include <type_traits>
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
  SparseMatrix(std::vector<Int> const& range, std::vector<Int> const& index)
      : range_(range), index_(index) {
  }

 public:
  const Int& range(Int i) const {
    return range_[i];
  }
  const Int& index(Int i) const {
    return index_[i];
  }
  Int CountVertices() const {
    return range_.size() - 1;
  }
  Int CountEdges() const {
    return index_.size();
  }
  Int& range(Int i) {
    return range_[i];
  }
  Int& index(Int i) {
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

template <typename Int>
class Mesh : private SparseMatrix<Int> {
  static_assert(std::is_integral_v<Int>, "Integral required.");
  using Base = SparseMatrix<Int>;
  Int n_nodes_;

 public:
  Mesh() = default;
  Mesh(std::vector<Int> const& range, std::vector<Int> const& index)
      : Base(range, index) {
    n_nodes_ = range.back();
  }

 public:
  const Int& range(Int i) const {
    return this->Base::range(i);
  }
  const Int& nodes(Int i) const {
    return this->index(i);
  }
  Int CountCells() const {
    return this->CountVertices();
  }
  Int CountNodes() const {
    return n_nodes_;
  }
  Int& range(Int i) {
    return this->Base::range(i);
  }
  Int& nodes(Int i) {
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
 * @brief A wrapper of `METIS_Free()`.
 * 
 */
auto deleter = [](void* p){ METIS_Free(p); };
using Deleter = decltype(deleter);

template <typename Int>
class SparseGraphWithDeleter {
  static_assert(std::is_integral_v<Int>, "Integral required.");
  Int size_, *range_, *index_/* , *value_ */;

 public:
  SparseGraphWithDeleter(Int size, Int* range, Int* index)
      : size_(size), range_(range), index_(index) {
  }
  ~SparseGraphWithDeleter() noexcept {
    int errors = (METIS_Free(range_) != METIS_OK)
               + (METIS_Free(index_) != METIS_OK);
    assert(errors == 0);
  }
  const Int& range(Int i) const {
    return range_[i];
  }
  const Int& index(Int i) const {
    return index_[i];
  }
  Int CountVertices() const {
    return size_;
  }
  Int CountEdges() const {
    return range_[size_];
  }
  Int& range(Int i) {
    return range_[i];
  }
  Int& index(Int i) {
    return index_[i];
  }
};

template <typename Int>
struct File {
  SparseMatrix<Int> cells;
};

}  // namespace metis
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_METIS_FORMAT_HPP_
