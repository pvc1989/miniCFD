// Copyright 2022 PEI Weicheng
#ifndef MINI_GEOMETRY_INTERSECT_HPP_
#define MINI_GEOMETRY_INTERSECT_HPP_

#include <concepts>

#include "mini/algebra/eigen.hpp"

namespace mini {
namespace geometry {

template <typename Point, std::floating_point Scalar>
void Intersect(Point const &pa, Point const &pb, Point const &pc,
    Point const &pq, Scalar *ratio) {
  mini::algebra::Matrix<Scalar, 3, 3> mat;
  mat.col(0) = pa;
  mat.col(1) = pb;
  mat.col(2) = pc;
  bool swap_pq = false;
  if (mat.determinant() == 0) {
    mat.col(0) -= pq;  // qa
    mat.col(1) -= pq;  // qb
    mat.col(2) -= pq;  // qc
    if (mat.determinant() == 0) {
      // TODO(PVC): pq on surface abc
    } else {
      swap_pq = true;
    }
  }
  Point lambda = mat.fullPivLu().solve(pq);
  if (swap_pq) {
    lambda = -lambda;
  }
  if (lambda.minCoeff() >= 0) {
    // The intersection of line PQ and triangle ABC is inside triangle ABC.
    auto sum = lambda.sum();
    if (sum >= 1.0) {
      // The intersection of line PQ and triangle ABC is on PQ.
      *ratio = (swap_pq ? sum - 1.0 : 1.0) / sum;
    }
  }
}

}  // namespace geometry
}  // namespace mini

#endif  // MINI_GEOMETRY_INTERSECT_HPP_
