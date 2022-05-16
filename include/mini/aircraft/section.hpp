// Copyright 2022 PEI Weicheng
#ifndef MINI_AIRCRAFT_SECTION_HPP_
#define MINI_AIRCRAFT_SECTION_HPP_

#include <cassert>
#include <iostream>
#include <utility>

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
  const Blade* blade_;
  std::pair<const Airfoil *, Scalar> left_, right_;
  Scalar y_ratio_, chord_, twist_/* deg */;

 public:
  Scalar GetAngleOfAttack(Scalar u, Scalar w) const {
    auto deg = mini::geometry::rad2deg(std::atan(w / u));  // [-90, 90]
    if (u < 0) {
      if (w > 0) {
        deg += 180/* [-90, 0] -> [90, 180] */;
      } else {
        deg -= 180/* [0, 90] -> [-180, -90] */;
      }
    }
    // deg := angle of inflow
    deg += GetTwist();
    // deg := angle of attack
    if (deg < -180 || 180 < deg) {
      deg += (deg < 0 ? 360 : -360);
    }
    // deg in [-180, 180]
    return deg;
  }

 public:
  Section(const Blade& blade, Scalar y_ratio, Scalar chord, Scalar twist,
      const Airfoil *a_0, Scalar w_0, const Airfoil *a_1, Scalar w_1)
      : blade_(&blade), y_ratio_(y_ratio), chord_(chord), twist_(twist),
        left_(a_0, w_0), right_(a_1, w_1) {
  }

  const Frame& GetFrame() const {
    return blade_->GetFrame();
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
    return blade_->GetPoint(y_ratio_);
  }
  Vector GetVelocity() const {
    auto omega = blade_->GetRotor().GetRadiansPerSecond();
    Vector v = -GetFrame().X();
    v *= omega * (blade_->GetRoot() + blade_->GetSpan() * y_ratio_);
    return v;
  }
  Vector GetForce(Scalar rho, Vector velocity) const {
    velocity -= GetVelocity();
    auto u = velocity.dot(GetFrame().X());
    auto w = velocity.dot(GetFrame().Z());
    auto deg = GetAngleOfAttack(u, w);
    Scalar c_lift = Lift(deg), c_drag = Drag(deg);
    deg -= GetTwist();  // angle of inflow
    auto [cos, sin] = mini::geometry::CosSin(deg);
    Scalar c_z = c_lift * cos + c_drag * sin;
    Scalar c_x = c_drag * cos - c_lift * sin;
    Vector force = c_z * GetFrame().Z() + c_x * GetFrame().X();
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
