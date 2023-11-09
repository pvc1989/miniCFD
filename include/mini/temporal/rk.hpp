// Copyright 2023 PEI Weicheng
#ifndef MINI_TEMPORAL_RK_HPP_
#define MINI_TEMPORAL_RK_HPP_

#include "mini/temporal/ode.hpp"

namespace mini {
namespace temporal {

template <int kOrders, typename Scalar>
struct RungeKutta;

template <typename Scalar>
struct RungeKutta<1, Scalar> : public Solver<Scalar> {
 private:
  using Base = Solver<Scalar>;

 public:
  using Column = typename Base::Column;

  void Update(System<Scalar> *system, double t_curr, double dt) final {
    auto u_next = Euler<Scalar>::NextSolution(system, t_curr, dt);
    system->SetSolutionColumn(u_next);
  }
};

}  // namespace temporal
}  // namespace mini

#endif  // MINI_TEMPORAL_RK_HPP_
