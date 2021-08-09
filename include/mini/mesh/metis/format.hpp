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
  std::vector<Int> const& range() const {
    return range_;
  }
  std::vector<Int> const& index() const {
    return index_;
  }
  Int CountVertices() const {
    return range_.size() - 1;
  }
  Int CountEdges() const {
    return index_.size();
  }
  std::vector<Int>& range() {
    return range_;
  }
  std::vector<Int>& index() {
    return index_;
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
  std::vector<Int> const& range() const {
    return this->Base::range();
  }
  std::vector<Int> const& nodes() const {
    return this->index();
  }
  Int CountCells() const {
    return this->CountVertices();
  }
  Int CountNodes() const {
    return n_nodes_;
  }
  std::vector<Int>& range() {
    return this->Base::range();
  }
  std::vector<Int>& nodes() {
    return this->index();
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
class SparseMatrixWithDeleter {
  static_assert(std::is_integral_v<Int>, "Integral required.");
  std::unique_ptr<Int[], Deleter> range_, index_/* , value_ */;
  Int size_;

 public:
  SparseMatrixWithDeleter(Int size, Int* range, Int* index)
      : size_(size), range_(range, deleter), index_(index, deleter) {
  }
  Int size() const {
    return size_;
  }
  auto const& range() const {
    return range_;
  }
  auto  const& index() const {
    return index_;
  }
  auto& range() {
    return range_;
  }
  auto& index() {
    return index_;
  }

  void reset(Int *range, Int *neighbors) {
    range_.reset(range);
    index_.reset(neighbors);
  }
};

template <typename Int>
class GraphWithDeleter : public SparseMatrixWithDeleter<Int> {
 private:
  using Base = SparseMatrixWithDeleter<Int>;

 public:
  using Base::Base;
};

template <typename Int>
struct File {
  SparseMatrix<Int> cells;
};

}  // namespace metis
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_METIS_FORMAT_HPP_
