// Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_RIEMANN_ROTATED_MULTIPLE_HPP_
#define MINI_RIEMANN_ROTATED_MULTIPLE_HPP_

#include "mini/riemann/rotated/simple.hpp"
#include "mini/riemann/linear/multiple.hpp"

namespace mini {
namespace riemann {
namespace rotated {

template <typename S, int K, int D>
class Multiple : public Simple<linear::Multiple<S, K, D>> {
  using Base = Simple<linear::Multiple<S, K, D>>;

 public:
  using Jacobi = typename Base::Jacobi;
  using Conservative = typename Base::Conservative;

  const Jacobi& L(const Conservative &) const {
    return this->unrotated_simple_.L();
  }
  const Jacobi& R(const Conservative &) const {
    return this->unrotated_simple_.R();
  }
};

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_ROTATED_MULTIPLE_HPP_
