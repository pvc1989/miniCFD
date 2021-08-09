// Copyright 2021 PEI Weicheng and YANG Minghao and JIANG Yuyan

#ifndef MINI_MESH_METIS_FORMAT_HPP_
#define MINI_MESH_METIS_FORMAT_HPP_

#include <type_traits>
#include <vector>

namespace mini {
namespace mesh {
namespace metis {

template <typename Int>
struct SparseMatrix {
  static_assert(std::is_integral_v<Int>, "Integral required.");
  std::vector<Int> range;
  std::vector<Int> index;
};

template <typename Int>
struct File {
  SparseMatrix<Int> cells;
};

}  // namespace metis
}  // namespace mesh
}  // namespace mini

#endif  // MINI_MESH_METIS_FORMAT_HPP_
