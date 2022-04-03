// Copyright 2022 PEI Weicheng
#ifndef MINI_WING_AIRFOIL_HPP_
#define MINI_WING_AIRFOIL_HPP_

namespace mini {
namespace wing {
namespace airfoil {

template <typename Scalar>
class Abstract {
 public:
  virtual Scalar Lift(Scalar alpha) const = 0;
  virtual Scalar Drag(Scalar alpha) const = 0;
};

template <typename Scalar>
class Simple : public Abstract<Scalar> {
  Scalar c_lift_, c_drag_;

 public:
  Simple(Scalar c_l, Scalar c_d)
      : c_lift_(c_l), c_drag_(c_d) {
  }
  Scalar Lift(Scalar alpha) const override {
    return c_lift_;
  }
  Scalar Drag(Scalar alpha) const override {
    return c_drag_;
  }
};

}  // namespace airfoil
}  // namespace wing
}  // namespace mini

#endif  // MINI_WING_AIRFOIL_HPP_
