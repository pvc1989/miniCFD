//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_INTEGRATOR_BASE_HPP_
#define MINI_INTEGRATOR_BASE_HPP_

#include <cmath>
#include <iostream>
#include <type_traits>

#include <Eigen/Dense>

namespace mini {
namespace integrator {

template <class Object>
void print(Object&& obj) {
  std::cout << obj << '\n' << std::endl;
}

template <typename Callable, typename Element>
auto Quadrature(Callable&& f_in_local, Element&& element) {
  using E = std::remove_reference_t<Element>;
  using LocalCoord = typename E::LocalCoord;
  decltype(f_in_local(LocalCoord())) sum; sum *= 0;
  for (int i = 0; i < E::CountQuadPoints(); ++i) {
    auto f_val = f_in_local(E::GetCoord(i));
    f_val *= E::GetWeight(i);
    sum += f_val;
  }
  return sum;
}

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

template <typename Func1, typename Func2, typename Element>
auto Innerprod(Func1&& f1, Func2&& f2, Element&& element) {
  using E = std::remove_reference_t<Element>;
  using GlobalCoord = typename E::GlobalCoord;
  return Integrate([&f1, &f2](const GlobalCoord& xyz_global){
    return f1(xyz_global) * f2(xyz_global);
  }, element);
}

template <typename Callable, typename Element>
auto Norm(Callable&& f, Element&& element) {
  return std::sqrt(Innerprod(f, f, element));
}

template <typename Scalar, int kDim, int kOrder>
class Basis;

template <typename Scalar>
class Basis<Scalar, 2, 2> {

 public:
  static constexpr int N = 6;
  using Mat2x1 = Eigen::Matrix<Scalar, 2, 1>;
  using MatNx1 = Eigen::Matrix<Scalar, N, 1>;
  using MatNxN = Eigen::Matrix<Scalar, N, N>;
  explicit Basis(Mat2x1 const& c = {0, 0})
      : center_(c) {
  }
  Basis(const Basis&) = default;
  Basis(Basis&&) noexcept = default;
  Basis& operator=(const Basis&) = default;
  Basis& operator=(Basis&&) noexcept = default;
  ~Basis() noexcept = default;

  MatNx1 operator()(Mat2x1 const& xy) const {
    auto x = xy[0] - center_[0], y = xy[1] - center_[1];
    MatNx1 col = { 1, x, y, x * x, x * y, y * y };
    return coef_ * col;
  }
  void Transform(MatNxN const& a) {
    coef_ = a;
  }

 private:
  Mat2x1 center_;
  MatNxN coef_ = MatNxN::Identity();
};

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

#endif  // MINI_INTEGRATOR_BASE_HPP_
