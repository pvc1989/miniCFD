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
 public:
  Scalar Lift(Scalar alpha) const override {
    return 6.0;
  }
  Scalar Drag(Scalar alpha) const override {
    return 0.0;
  }
};

}  // namespace airfoil
}  // namespace wing
}  // namespace mini

#endif  // MINI_WING_AIRFOIL_HPP_
