// Copyright 2022 PEI Weicheng
#ifndef MINI_WING_ROTARY_HPP_
#define MINI_WING_ROTARY_HPP_

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
class Rotor;

template <typename Scalar>
class Blade {
  using Airfoil = mini::wing::Airfoil<Scalar>;
  using Rotor = mini::wing::Rotor<Scalar>;
  using Frame = mini::geometry::Frame<Scalar>;
  using Vector = typename Frame::Vector;
  using Point = Vector;

  std::vector<Scalar> positions_, chords_, twists_;
  std::vector<Airfoil> airfoils_;
  Frame frame_;  // absolute
  Scalar root_, azimuth_;
  const Rotor* rotor_;

  int CountSections() const {
    return positions_.size();
  }

 public:
  Blade& SetRotor(const Rotor& rotor) {
    rotor_ = &rotor;
    return *this;
  }
  Blade& SetRoot(Scalar root) {
    root_ = root;
    return *this;
  }
  Blade& SetAzimuth(Scalar deg) {
    azimuth_ = deg;
    frame_ = GetRotor().GetFrame();
    frame_.RotateZ(deg);
    return *this;
  }
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
  Scalar GetSpan() const {
    return positions_.empty() ? 0.0 : positions_.back();
  }

  /**
   * @brief Get the `Frame` of a `Blade`.
   * 
   * @return const Frame& The absolute orientation of a `Frame`.
   */
  const Frame& GetFrame() const {
    return frame_;
  }
  const Rotor& GetRotor() const {
    return *rotor_;
  }

  /**
   * @brief Get the origin of the `Frame` of a `Blade`.
   * 
   * @return Point The absolute coordinates of a `Point`.
   */
  Point GetOrigin() const {
    Point point = GetRoot() * GetFrame().Y();
    point += GetRotor().GetOrigin();
    return point;
  }
  Scalar GetRoot() const {
    return root_;
  }
  Scalar GetAzimuth() const {
    return azimuth_;
  }

  /**
   * @brief Get a `Point` on the span axis of this `Blade`.
   * 
   * @param y_blade The dimensionless position, `0` for root, `1` for tip.
   * @return Point The absolute coordinates of the `Point`.
   */
  Point GetPoint(Scalar y_blade) const {
    if (y_blade < 0 || 1 < y_blade) {
      throw std::range_error("The dimensionless position must be in [0.0, 1.0].");
    }
    Point point = GetFrame().Y();
    point *= y_blade * GetSpan();
    point += GetOrigin();
    return point;
  }
};

template <typename Scalar>
class Rotor {
  using Frame = mini::geometry::Frame<Scalar>;
  using Blade = mini::wing::Blade<Scalar>;
  using Vector = typename Frame::Vector;
  using Point = Vector;

  std::vector<Blade> blades_;
  Frame frame_;
  Point origin_;
  Scalar azimuth_/* deg */, omega_/* deg / s */;

 public:
  Rotor& SetOmega(Scalar rps) {
    omega_ = rps * 360.0;
    return *this;
  }

  Rotor& SetOrigin(Scalar x, Scalar y, Scalar z) {
    origin_ << x, y, z;
    return *this;
  }

  Rotor& SetFrame(const Frame& frame) {
    frame_ = frame;
    return *this;
  }

  Rotor& SetAzimuth(Scalar deg) {
    azimuth_ = deg;
    auto psi = 360.0 / CountBlades();
    for (auto &blade : blades_) {
      blade.SetAzimuth(deg);
      deg += psi;
    }
    return *this;
  }

  Rotor& InstallBlade(Scalar root, const Blade& blade) {
    blades_.emplace_back(blade);
    blades_.back().SetRoot(root).SetRotor(*this);
    return *this;
  }

 public:
  int CountBlades() const {
    return blades_.size();
  }

  /**
   * @brief Get the rotating speed of this `Rotor`.
   * 
   * @return Scalar The rotating speed in degrees per second.
   */
  Scalar GetOmega() const {
    return omega_;
  }

  /**
   * @brief Get the azimuth of this `Rotor`.
   * 
   * @return Scalar The azimuth in degrees.
   */
  Scalar GetAzimuth() const {
    return azimuth_;
  }

  const Point& GetOrigin() const {
    return origin_;
  }

  const Frame& GetFrame() const {
    return frame_;
  }

  const Blade& GetBlade(int i) const {
    return blades_.at(i);
  }
};

}  // namespace wing
}  // namespace mini

#endif  // MINI_WING_ROTARY_HPP_
