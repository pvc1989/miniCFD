//  Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_RIEMANN_LINEAR_SINGLE_HPP_
#define MINI_RIEMANN_LINEAR_SINGLE_HPP_

#include <cmath>
#include <array>

#include "mini/algebra/column.hpp"
#include "mini/algebra/matrix.hpp"

namespace mini {
namespace riemann {
namespace linear {

class SingleWave {
 public:
  // Types:
  using Jacobi = double;
  using State = double;
  using Flux = double;
  using Speed = double;
  // Constructor:
  SingleWave() : a_const_(1) {}
  explicit SingleWave(Jacobi const& a_const) : a_const_(a_const) {}
  // Get F on T Axia
  Flux GetFluxOnTimeAxis(State const& left, State const& right) const {
    if (0 < a_const_) {
      return left * a_const_;
    } else {
      return right* a_const_;
    }
  }
  // Get F of U
  Flux GetFlux(State const& state) const { return state * a_const_ ; }

 private:
  Jacobi a_const_;
};

}  // namespace linear
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_LINEAR_SINGLE_HPP_
