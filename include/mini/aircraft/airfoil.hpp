// Copyright 2022 PEI Weicheng
#ifndef MINI_AIRCRAFT_AIRFOIL_HPP_
#define MINI_AIRCRAFT_AIRFOIL_HPP_

#include <cmath>

namespace mini {
namespace aircraft {
namespace airfoil {

template <typename Scalar>
class Abstract {
 public:
  virtual Scalar Lift(Scalar deg) const = 0;
  virtual Scalar Drag(Scalar deg) const = 0;
};

template <typename Scalar>
class Linear : public Abstract<Scalar> {
  Scalar c_lift_, c_drag_;

 public:
  Linear(Scalar c_lift/* deg^{-1} */, Scalar c_drag/* deg^{-1} */)
      : c_lift_(c_lift), c_drag_(c_drag) {
  }
  Scalar Lift(Scalar deg) const override {
    return c_lift_ * deg;
  }
  Scalar Drag(Scalar deg) const override {
    return c_drag_ * std::abs(deg);
  }
};

}  // namespace airfoil
}  // namespace aircraft
}  // namespace mini

#endif  // MINI_AIRCRAFT_AIRFOIL_HPP_
