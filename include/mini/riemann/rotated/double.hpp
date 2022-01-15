// Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_RIEMANN_ROTATED_DOUBLE_HPP_
#define MINI_RIEMANN_ROTATED_DOUBLE_HPP_

#include "mini/riemann/rotated/simple.hpp"
#include "mini/riemann/linear/double.hpp"

namespace mini {
namespace riemann {
namespace rotated {

template <int kDim>
class Double : public Simple<linear::Double<kDim>> {
  using Base = Simple<linear::Double<kDim>>;

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

#endif  // MINI_RIEMANN_ROTATED_DOUBLE_HPP_
