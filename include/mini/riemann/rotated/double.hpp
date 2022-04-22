// Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_RIEMANN_ROTATED_DOUBLE_HPP_
#define MINI_RIEMANN_ROTATED_DOUBLE_HPP_

#include "mini/riemann/rotated/simple.hpp"
#include "mini/riemann/linear/double.hpp"

namespace mini {
namespace riemann {
namespace rotated {

template <typename S, int D>
class Double : public Simple<linear::Double<S, D>> {
  using Base = Simple<linear::Double<S, D>>;

 public:
  constexpr static int kComponents = 2;
  constexpr static int kDimensions = D;
  using Scalar = S;
  using Jacobi = typename Base::Jacobi;
  using Conservative = typename Base::Conservative;

  void UpdateEigenMatrices(const Conservative &) {
  }
  const Jacobi& L() const {
    return this->unrotated_simple_.L();
  }
  const Jacobi& R() const {
    return this->unrotated_simple_.R();
  }
};

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_ROTATED_DOUBLE_HPP_
