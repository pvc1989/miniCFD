//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_INTEGRATOR_FUNCTION_HPP_
#define MINI_INTEGRATOR_FUNCTION_HPP_

#include <cmath>
#include <iostream>
#include <type_traits>

#include "mini/algebra/eigen.hpp"

namespace mini {
namespace integrator {

template <class Object>
void print(Object&& obj) {
  std::cout << obj << '\n' << std::endl;
}

/**
 * @brief Set the value of a scalar to be 0.
 * 
 * @tparam Scalar the type of the scalar
 * @param s the address of the scalar
 */
template <class Scalar>
inline void SetZero(Scalar* s) {
  static_assert(std::is_scalar_v<Scalar>);
  *s = 0;
}

/**
 * @brief Set all coefficients of a matrix to be 0.
 * 
 * @tparam Scalar the type of the matrix's coefficient
 * @tparam M the number of rows of the matrix
 * @tparam N the number of columns of the matrix
 * @param m the address of the matrix
 */
template <class Scalar, int M, int N>
inline void SetZero(algebra::Matrix<Scalar, M, N>* m) {
  m->setZero();
}

/**
 * @brief Perform Gaussian quadrature of a callable object on an integratable element in the parametric space.
 * 
 * @tparam Callable the type of the integrand
 * @tparam Element the type of the integrator
 * @param f_in_local the integrand using local coordinates as arguments
 * @param element the integrator
 * @return auto the value of the integral
 */
template <typename Callable, typename Element>
auto Quadrature(Callable&& f_in_local, Element&& element) {
  using E = std::remove_reference_t<Element>;
  using LocalCoord = typename E::LocalCoord;
  decltype(f_in_local(LocalCoord())) sum{};
  SetZero(&sum);
  for (int i = 0; i < E::CountQuadPoints(); ++i) {
    auto f_val = f_in_local(E::GetCoord(i));
    f_val *= E::GetWeight(i);
    sum += f_val;
  }
  return sum;
}

/**
 * @brief Perform Gaussian quadrature of a callable object on an integratable element in the physical space.
 * 
 * @tparam Callable the type of the integrand
 * @tparam Element the type of the integrator
 * @param f_in_global the integrand using global coordinates as arguments
 * @param element the integrator
 * @return auto the value of the integral
 */
template <typename Callable, typename Element>
auto Integrate(Callable&& f_in_global, Element&& element) {
  using E = std::remove_reference_t<Element>;
  using LocalCoord = typename E::LocalCoord;
  auto f_in_local = [&element, &f_in_global](const LocalCoord& xyz_local) {
    auto f_val = f_in_global(element.local_to_global_Dx1(xyz_local));
    auto mat_j = element.jacobian(xyz_local);
    auto det_j = E::CellDim() < E::PhysDim()
        ? (mat_j.transpose() * mat_j).determinant()
        : mat_j.determinant();
    f_val *= std::sqrt(det_j);
    return f_val;
  };
  return Quadrature(f_in_local, element);
}

/**
 * @brief Calculate the inner-product of two functions on an integratable element.
 * 
 * @tparam Func1 the type of the first function
 * @tparam Func2 the type of the second function
 * @tparam Element the type of the integrator
 * @param f1 the first function
 * @param f2 the second function
 * @param element the integrator
 * @return auto the value of the innerproduct
 */
template <typename Func1, typename Func2, typename Element>
auto Innerprod(Func1&& f1, Func2&& f2, Element&& element) {
  using E = std::remove_reference_t<Element>;
  using GlobalCoord = typename E::GlobalCoord;
  return Integrate([&f1, &f2](const GlobalCoord& xyz_global){
    return f1(xyz_global) * f2(xyz_global);
  }, element);
}

/**
 * @brief Calculate the 2-norm of a function on an integratable element.
 * 
 * @tparam Callable type of the function
 * @tparam Element the type of the integrator
 * @param f the function
 * @param element the integrator
 * @return auto the value of the norm
 */
template <typename Callable, typename Element>
auto Norm(Callable&& f, Element&& element) {
  return std::sqrt(Innerprod(f, f, element));
}

/**
 * @brief Change a group of linearly independent functions into an orthonormal basis.
 * 
 * @tparam Basis the type of the basis
 * @tparam Element the type of the integrator
 * @param raw_basis the raw basis whose components are linearly independent from each other
 * @param elem the integrator
 */
template <class Basis, class Element>
void Orthonormalize(Basis* raw_basis, const Element& elem) {
  constexpr int N = Basis::N;
  using MatNxN = typename Basis::MatNxN;
  using MatDx1 = typename Element::GlobalCoord;
  using Scalar = typename Element::Real;
  MatNxN S; S.setIdentity();
  auto A = Integrate([raw_basis](const MatDx1& xyz){
    auto col = (*raw_basis)(xyz);
    MatNxN result = col * col.transpose();
    return result;
  }, elem);
  S(0, 0) = 1 / std::sqrt(A(0, 0));
  for (int i = 1; i < N; ++i) {
    for (int j = 0; j < i; ++j) {
      Scalar temp = 0;
      for (int k = 0; k <= j; ++k) {
        temp += S(j, k) * A(k, i);
      }
      for (int l = 0; l <= j; ++l) {
        S(i, l) -= temp * S(j, l);
      }
    }
    Scalar norm_sq = 0;
    for (int j = 0; j <= i; ++j) {
      Scalar sum = 0, Sij = S(i, j);
      for (int k = 0; k < j; ++k) {
        sum += 2 * S(i, k) * A(k, j);
      }
      norm_sq += Sij * (Sij * A(j, j) + sum);
    }
    S.row(i) /= std::sqrt(norm_sq);
  }
  raw_basis->Transform(S);
}

}  // namespace integrator
}  // namespace mini

#endif  // MINI_INTEGRATOR_FUNCTION_HPP_
