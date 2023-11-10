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
  Euler<Scalar> euler_;

 public:
  using Column = typename Base::Column;

  void Update(System<Scalar> *system, double t_curr, double dt) final {
    euler_.Update(system, t_curr, dt);
  }
};

template <typename Scalar>
struct RungeKutta<2, Scalar> : public Solver<Scalar> {
 private:
  using Base = Solver<Scalar>;
  Euler<Scalar> euler_;

 public:
  using Column = typename Base::Column;

  void Update(System<Scalar> *system, double t_curr, double dt) final {
    auto u_curr = system->GetSolutionColumn();
    euler_.Update(system, t_curr, dt);
    auto u_next = euler_.NextSolution(system, t_curr + dt, dt);
    u_next += u_curr;
    u_next *= 0.5;
    system->SetSolutionColumn(u_next);
  }
};

template <typename Scalar>
struct RungeKutta<3, Scalar> : public Solver<Scalar> {
 private:
  using Base = Solver<Scalar>;
  Euler<Scalar> euler_;

 public:
  using Column = typename Base::Column;

  void Update(System<Scalar> *system, double t_curr, double dt) final {
    auto u_curr = system->GetSolutionColumn();
    euler_.Update(system, t_curr, dt);
    // Now, system->GetSolutionColumn() == U_1st == U_old + R_old * dt
    auto u_next = u_curr;
    u_next *= 3;
    u_next += euler_.NextSolution(system, t_curr + dt, dt);
    u_next /= 4;
    // Now, u_next == U_2nd == ((U_1st + R_1st * dt) + U_old * 3) / 4
    system->SetSolutionColumn(u_next);
    u_next = euler_.NextSolution(system, t_curr + dt / 2, dt);
    // Now, u_next == U_2nd + R_2nd * dt
    u_next *= 2;
    u_next += u_curr;
    u_next /= 3;
    // Now, u_next == U_3rd == ((U_2nd + R_2nd * dt) * 2 + U_old) / 3
    system->SetSolutionColumn(u_next);
  }
};

}  // namespace temporal
}  // namespace mini

#endif  // MINI_TEMPORAL_RK_HPP_
