// Copyright 2021 PEI Weicheng and JIANG Yuyan
/**
 * This file defines parser of partition info txt.
 */
#ifndef MINI_ALGEBRA_EIGEN_HPP_
#define MINI_ALGEBRA_EIGEN_HPP_

#include "Eigen/Eigen"

namespace mini {
namespace algebra {

using Eigen::Array;
using Eigen::Matrix;
using Eigen::Vector;

template <class MatrixType>
using LowerTriangularView = Eigen::TriangularView<MatrixType, Eigen::Lower>;

template <class MatrixType>
auto GetLowerTriangularView(MatrixType const &matrix) {
  return matrix.template triangularView<Eigen::Lower>();
}
template <class MatrixType>
auto GetLowerTriangularView(MatrixType *matrix) {
  return matrix->template triangularView<Eigen::Lower>();
}

}  // namespace algebra
}  // namespace mini

#endif  // MINI_ALGEBRA_EIGEN_HPP_
