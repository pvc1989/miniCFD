// Copyright 2022 PEI Weicheng
#ifndef MINI_TEMPORAL_RUNGE_KUTTA_HPP_
#define MINI_TEMPORAL_RUNGE_KUTTA_HPP_

template <typename SemiDiscreteSystem>
class RungeKuttaBase {
};

template <int kOrders, typename SemiDiscreteSystem>
struct RungeKutta;

template <typename SemiDiscreteSystem>
struct RungeKutta<1, SemiDiscreteSystem>
    : public RungeKuttaBase<SemiDiscreteSystem> {
 private:
  using Base = RungeKuttaBase<SemiDiscreteSystem>;

 public:
  using Base::Base;
  RungeKutta(const RungeKutta &) = default;
  RungeKutta &operator=(const RungeKutta &) = default;
  RungeKutta(RungeKutta &&) noexcept = default;
  RungeKutta &operator=(RungeKutta &&) noexcept = default;
  ~RungeKutta() noexcept = default;

 public:
  void Update(SemiDiscreteSystem *sds, double t_curr, double dt) {
  }
};

#endif  // MINI_TEMPORAL_RUNGE_KUTTA_HPP_
