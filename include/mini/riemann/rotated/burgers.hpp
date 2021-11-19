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
  using Base = Simple<nonlinear::Burgers<kDim>>;
  static constexpr int K = Base::K; static_assert(K == 1);
  static constexpr int D = Base::D;

 public:
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using State = typename Base::State;
  using Flux = typename Base::Flux;
  using FluxMatrix = typename Base::FluxMatrix;

  static FluxMatrix GetFluxMatrix(const State& state) {
    FluxMatrix flux_mat;
    for (int c = 0; c < D; ++c) {
      flux_mat(0, c) = state[0] * state[0];
      flux_mat(0, c) *= Base::global_coefficient[c] / 2.0;
    }
    return flux_mat;
  }
};

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_ROTATED_BURGERS_HPP_
