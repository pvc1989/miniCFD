// Copyright 2021 PEI Weicheng and YANG Minghao and JIANG Yuyan

#ifndef MINI_MESH_METIS_FORMAT_HPP_
#define MINI_MESH_METIS_FORMAT_HPP_

#include <memory>
#include <type_traits>
#include <vector>

#include "metis.h"

namespace mini {
namespace mesh {
namespace metis {

/**
 * @brief The data structure for storing sparse graphs or meshes.
 * 
 * In a real CSR (Compressed Sparse Row) matrix, the column indices for row `i` are stored in `index_[ range_[i], range_[i+1] )` and their corresponding values are stored in `value_[ range_[i], range_[i+1] )`.
 * To represent a graph of mesh, We only care about the positions of non-zero values, so we have ignored the `value_` array.
 * 
 * @tparam Int index type
 */
template <typename Int>
struct SparseMatrix {
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
  std::vector<Int>& range() {
    return range_;
  }
  std::vector<Int>& index() {
    return index_;
  }
  void resize(int n_rows, int n_nonzeros) {
    range_.resize(n_rows + 1);
    index_.resize(n_nonzeros);
  }
};

template <typename Int>
struct Mesh : public SparseMatrix<Int> {
  static_assert(std::is_integral_v<Int>, "Integral required.");

 public:
  std::vector<Int> const& nodes() const {
    return this->index();
  }
  std::vector<Int>& nodes() {
    return this->index();
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
