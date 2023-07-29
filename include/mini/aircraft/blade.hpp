// Copyright 2022 PEI Weicheng
#ifndef MINI_AIRCRAFT_BLADE_HPP_
#define MINI_AIRCRAFT_BLADE_HPP_

#include <concepts>

#include <iostream>
#include <stdexcept>
#include <vector>

#include "mini/aircraft/section.hpp"

namespace mini {
namespace aircraft {

template <std::floating_point Scalar>
class Rotor;

// TODO(PVC): Support swept blades.
template <std::floating_point Scalar>
class Blade {
 public:
  using Rotor = mini::aircraft::Rotor<Scalar>;
  using Section = mini::aircraft::Section<Scalar>;
  using Airfoil = typename Section::Airfoil;
  using Frame = typename Section::Frame;
  using Vector = typename Section::Vector;

 private:
  std::vector<Scalar> y_values_, chords_, twists_;
  // pointers for polymorphism; resources managed by some higher object:
  std::vector<const Airfoil *> airfoils_;
  Vector p_, q_, pq_;
  Frame frame_;  // absolute
  Scalar root_, azimuth_;
  const Rotor *rotor_;

 public:
  Blade &SetRotor(const Rotor &rotor) {
    rotor_ = &rotor;
    return *this;
  }
  Blade &SetRoot(Scalar root) {
    root_ = root;
    return *this;
  }
  Blade &SetAzimuth(Scalar deg) {
    azimuth_ = deg;
    frame_ = GetRotor().GetFrame();
    frame_.RotateZ(deg);
    p_ = GetPoint(0);
    q_ = GetPoint(1);
    pq_ = q_ - p_;
    return *this;
  }
  Blade &InstallSection(Scalar y_value, Scalar chord, Scalar twist,
      const Airfoil &airfoil) {
    if (CountSections() == 0 && y_value) {
      throw std::invalid_argument("First position must be 0.");
    }
    if (CountSections() && y_values_.back() >= y_value) {
      throw std::invalid_argument("New position must > existing ones.");
    }
    y_values_.emplace_back(y_value);
    chords_.emplace_back(chord);
    twists_.emplace_back(twist);
    airfoils_.emplace_back(&airfoil);
    return *this;
  }

  int CountSections() const {
    return y_values_.size();
  }
  Scalar GetSpan() const {
    return y_values_.empty() ? 0.0 : y_values_.back();
  }

  /**
   * @brief Get the `Frame` of a `Blade`.
   * 
   * @return const Frame & The absolute orientation of a `Frame`.
   */
  const Frame &GetFrame() const {
    return frame_;
  }
  const Rotor &GetRotor() const {
    return *rotor_;
  }

  /**
   * @brief Get the origin of the `Frame` of a `Blade`.
   * 
   * @return Vector The absolute coordinates of a point.
   */
  Vector GetOrigin() const {
    Vector point = GetRoot() * GetFrame().Y();
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
   * @brief Get a point on the span axis of this `Blade`.
   * 
   * @param y_ratio The dimensionless position, `0` for root, `1` for tip.
   * @return Vector The absolute coordinates of the point.
   */
  Vector GetPoint(Scalar y_ratio) const {
    if (y_ratio < 0 || 1 < y_ratio) {
      throw std::domain_error("The argument must be in [0.0, 1.0].");
    }
    Vector point = GetFrame().Y();
    point *= y_ratio * GetSpan();
    point += GetOrigin();
    return point;
  }
  const Vector &P() const {
    return p_;
  }
  const Vector &Q() const {
    return q_;
  }
  const Vector &PQ() const {
    return pq_;
  }

  Section GetSection(Scalar y_ratio) const {
    assert(0 <= y_ratio && y_ratio <= 1);
    auto y_value = y_ratio * GetSpan();
    auto itr = std::lower_bound(y_values_.begin(), y_values_.end(), y_value);
    assert(itr != y_values_.end());
    auto i_1 = itr - y_values_.begin();
    auto chord = chords_[i_1];
    auto twist = twists_[i_1];
    auto a_1 = airfoils_[i_1];
    auto a_0 = a_1;
    auto w_1 = 1.0, w_0 = 0.0;
    if (*itr != y_value) {
      // linear interpolation
      assert(i_1 > 0);
      auto i_0 = i_1 - 1;
      auto lambda_1 = (itr[0] - y_value) / (itr[0] - itr[-1]);
      auto lambda_0 = 1 - lambda_1;
      chord = chord * lambda_1 + chords_[i_0] * lambda_0;
      twist = twist * lambda_1 + twists_[i_0] * lambda_0;
      a_0 = airfoils_[i_0];
      w_0 = lambda_0;
      w_1 = lambda_1;
    }
    return Section(*this, y_ratio, chord, twist, a_0, w_0, a_1, w_1);
  }
};

}  // namespace aircraft
}  // namespace mini

#endif  // MINI_AIRCRAFT_BLADE_HPP_
