// Copyright 2022 PEI Weicheng
#ifndef MINI_AIRCRAFT_SECTION_HPP_
#define MINI_AIRCRAFT_SECTION_HPP_

#include <iostream>

#include "mini/aircraft/airfoil.hpp"
#include "mini/geometry/frame.hpp"

namespace mini {
namespace aircraft {

template <typename Scalar>
class Blade;

template <typename Scalar>
class Section {
 public:
  using Blade = mini::aircraft::Blade<Scalar>;
  using Airfoil = mini::aircraft::airfoil::Abstract<Scalar>;
  using Frame = mini::geometry::Frame<Scalar>;
  using Vector = typename Frame::Vector;

 private:
  Frame frame_;
  const Blade* blade_;
  std::pair<const Airfoil *, Scalar> left_, right_;
  Scalar y_ratio_, chord_, twist_/* deg */;

 public:
  Section(const Blade& blade, Scalar y_ratio, Scalar chord, Scalar twist,
      const Airfoil *a_0, Scalar w_0, const Airfoil *a_1, Scalar w_1)
      : blade_(&blade), y_ratio_(y_ratio), chord_(chord), twist_(twist),
        left_(a_0, w_0), right_(a_1, w_1), frame_(blade.GetFrame()) {
    frame_.RotateY(twist);
  }

  const Blade& GetBlade() const {
    return *blade_;
  }
  const Frame& GetFrame() const {
    return frame_;
  }
  Scalar GetChord() const {
    return chord_;
  }
  Scalar GetTwist() const {
    return twist_;
  }
  Scalar Lift(Scalar deg) const {
    return left_.first->Lift(deg) * left_.second
        + right_.first->Lift(deg) * right_.second;
  }
  Scalar Drag(Scalar deg) const {
    return left_.first->Drag(deg) * left_.second
        + right_.first->Drag(deg) * right_.second;
  }
  Vector GetOrigin() const {
    return GetBlade().GetPoint(y_ratio_);
  }
  Vector GetVelocity() const {
    auto& blade = GetBlade();
    auto omega = blade.GetRotor().GetOmega();
    Vector v = -blade.GetFrame().X();
    v *= omega * (blade.GetRoot() + blade.GetSpan() * y_ratio_);
    return v;
  }
  Vector GetForce(Scalar rho, Vector velocity) const {
    velocity -= GetVelocity();
    auto u = velocity.dot(frame_.X());
    auto w = velocity.dot(frame_.Z());
    auto deg = mini::geometry::rad2deg(std::atan(w / u));
    Vector force = Lift(deg) * frame_.Z() + Drag(deg) * frame_.X();
    force *= -0.5 * rho * (u * u + w * w) * GetChord();
    return force;
  }
};

template <typename Scalar>
std::ostream& operator<<(std::ostream &out, const Section<Scalar> &section) {
  out << "frame = " << section.GetFrame() << "\n";
  out << "origin = " << section.GetOrigin().transpose() << "\n";
  out << "velocity = " << section.GetVelocity().transpose() << "\n";
  out << "chord = " << section.GetChord() << ", ";
  out << "twist = " << section.GetTwist() << " deg";
  return out;
}

}  // namespace aircraft
}  // namespace mini

#endif  // MINI_AIRCRAFT_SECTION_HPP_
