// Copyright 2022 PEI Weicheng
#ifndef MINI_WING_ROTARY_HPP_
#define MINI_WING_ROTARY_HPP_

#include <array>
#include <cmath>
#include <vector>
#include <stdexcept>

#include "mini/geometry/frame.hpp"

namespace mini {
namespace wing {

template <typename Scalar>
class Airfoil {
 public:
  Scalar Lift(Scalar alpha) const {
    return 6.0;
  }
  Scalar Drag(Scalar alpha) const {
    return 0.0;
  }
};

template <typename Scalar>
class Blade {
  using Airfoil = mini::wing::Airfoil<Scalar>;

  std::vector<Scalar> positions_, chords_, twists_;
  std::vector<Airfoil> airfoils_;

  int CountSections() const {
    return positions_.size();
  }

 public:
  Blade& InstallSection(Scalar position, Scalar chord, Scalar twist,
      const Airfoil& airfoil) {
    if (CountSections() && positions_.back() >= position) {
      throw std::range_error("New position must be larger than existing ones.");
    }
    positions_.emplace_back(position);
    chords_.emplace_back(chord);
    twists_.emplace_back(twist);
    airfoils_.emplace_back(airfoil);
    return *this;
  }
};

template <typename Scalar, int kBlades>
class Rotor {
  using Frame = mini::geometry::Frame<Scalar>;
  using Blade = mini::wing::Blade<Scalar>;
  using Vector = typename Frame::Vector;

  std::array<Blade, kBlades> blades_;
  std::array<Scalar, kBlades> roots_;
  Frame frame_;
  Vector origin_;
  Scalar omega_;

 public:
  Rotor& SetOmega(Scalar omega) {
    omega_ = omega;
    return *this;
  }
  Rotor& SetOrigin(Scalar x, Scalar y, Scalar z) {
    origin_ << x, y, z;
    return *this;
  }
  Rotor& SetOrientation(const Frame& frame) {
    frame_ = frame;
    return *this;
  }
  Rotor& InstallBlade(int i, Scalar root, const Blade& blade) {
    roots_.at(i) = root;
    blades_.at(i) = blade;
    return *this;
  }

 public:
  Scalar Azimuth(Scalar t) const {
    return omega_ * t;
  }

};

}  // namespace wing
}  // namespace mini

#endif  // MINI_WING_ROTARY_HPP_
