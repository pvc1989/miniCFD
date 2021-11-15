// Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_RIEMANN_ROTATED_BURGERS_HPP_
#define MINI_RIEMANN_ROTATED_BURGERS_HPP_

#include "mini/riemann/rotated/simple.hpp"
#include "mini/riemann/nonlinear/burgers.hpp"

namespace mini {
namespace riemann {
namespace rotated {

template <int kDim = 2>
class Burgers : public Simple<nonlinear::Burgers<kDim>> {
};

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_ROTATED_BURGERS_HPP_
