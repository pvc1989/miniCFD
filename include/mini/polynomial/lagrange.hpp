//  Copyright 2023 PEI Weicheng
#ifndef MINI_POLYNOMIAL_LAGRANGE_HPP_
#define MINI_POLYNOMIAL_LAGRANGE_HPP_

#include <concepts>

#include <cassert>
#include <cmath>

#include <array>
#include <initializer_list>

#include "mini/algebra/eigen.hpp"

#include "mini/polynomial/taylor.hpp"


namespace mini {
namespace polynomial {
namespace lagrange {

/**
 * @brief The Lagrange basis functions on the standard Line element.
 * 
 * The Lagrange polynomal at the k-th node \f$ L_k(\xi) \f$ is defined as \f$ \prod_{l=0,l\ne k}^{N-1}\frac{\xi-\xi_{l}}{\xi_{k}-\xi_{l}} \f$.
 * 
 * @tparam Scalar the type of nodes
 * @tparam kDegree the degree of each Lagrange polynomial
 */
template <std::floating_point Scalar, int kDegree>
class Line {
 public:
  static constexpr int P = kDegree;  // the degree of each member in this basis
  static constexpr int N = P + 1;  // the number of terms in this basis
  using Vector = algebra::Matrix<Scalar, 1, N>;
  using Matrix = algebra::Matrix<Scalar, N, N>;
  using Taylor = polynomial::Taylor<Scalar, 1, kDegree>;

 private:
  std::array<std::array<Vector, N>, N> derivatives_;  // derivatives_[k][j][i] := k-th order derivatives at the j-th node of the i-th basis
  Vector nodes_;  // nodes_[j] := coorinate of the j-th node 
  Matrix lagrange_to_taylor_;

 public:
  Line(std::initializer_list<Scalar> nodes) : nodes_{nodes} {
    assert(nodes.size() == N);
    /* An arbitrary polynomial can be expanded on Taylor and Lagrange:
     *     polynomial(x) = taylor_basis_row(x) @ taylor_coeff_col
     *                   = lagrange_basis_row(x) @ lagrange_coeff_col.
     * We want to find `taylor_to_lagrange` and `lagrange_to_taylor` such that
     *     taylor_coeff_col = lagrange_to_taylor @ lagrange_coeff_col,
     * and lagrange_coeff_col = taylor_to_lagrange @ taylor_coeff_col.
     * Then, from
     *     taylor_basis_row(x) @             taylor_coeff_col            = 
     *     taylor_basis_row(x) @ lagrange_to_taylor @ lagrange_coeff_col =
     *                   lagrange_basis_row(x)      @ lagrange_coeff_col,
     * We have
     *     taylor_basis_row(x) @ lagrange_to_taylor = lagrange_basis_row(x),
     * and lagrange_basis_row(x) @ taylor_to_lagrange = taylor_basis_row(x).
     * Since lagrange_basis_row(x[j])[i] = delta[i][j], we have
     *     taylor_to_lagrange[j] := values of Taylor basis at the j-th node.
     */
    Matrix taylor_to_lagrange;
    for (int j = 0; j < N; ++j) {
      auto x_j = nodes_[j];
      taylor_to_lagrange.row(j) = Taylor::GetValues(x_j);
    }
    lagrange_to_taylor_ = taylor_to_lagrange.inverse();
    assert((taylor_to_lagrange * lagrange_to_taylor_ - Matrix::Identity()).norm() < 1e-13);
    for (int k = 0; k < N; ++k) {
      for (int j = 0; j < N; ++j) {
        auto x_j = nodes_[j];
        derivatives_[k][j] = GetDerivatives(x_j, k);
      }
    }
  }

  /**
   * @brief Get the coordinate of the i-th node.
   * 
   * @param i the (0-based) index of the query node
   * @return Scalar the coordinate
   */
  Scalar GetNode(int i) const {
    return nodes_[i];
  }

  /**
   * @brief Get the values of all basis functions at an arbitrary point.
   * 
   * @param x the coordinate of the query point
   * @return Vector the values
   */
  Vector GetValues(Scalar x) const {
    Vector vec; vec.setOnes();
    for (int i = 0; i < N; ++i) {
      auto x_i = nodes_[i];
      for (int j = 0; j < N; ++j) {
        if (i != j) {
          auto x_j = nodes_[j];
          vec[i] *= (x - x_j) / (x_i - x_j);
        }
      }
    }
    return vec;
  }

  /**
   * @brief Get the k-th order derivatives of all basis functions at an arbitrary point.
   * 
   * @param x the coordinate of the query point
   * @param k the order of the derivatives to be taken
   * @return Vector the derivatives
   */
  Vector GetDerivatives(Scalar x, int k) const {
    Vector taylor_basis_row = Taylor::GetDerivatives(x, k);
    return taylor_basis_row * lagrange_to_taylor_;
  }

  /**
   * @brief Get the k-th order derivatives of all basis functions at the j-th node.
   * 
   * @param k the order of the derivatives to be taken
   * @return Vector the derivatives
   */
  Vector const &GetDerivatives(int k, int j) const {
    return derivatives_[k][j];
  }
};

}  // namespace lagrange
}  // namespace polynomial
}  // namespace mini

#endif  // MINI_POLYNOMIAL_LAGRANGE_HPP_
