// Copyright 2022 PEI Weicheng
#ifndef MINI_AIRCRAFT_ROTOR_HPP_
#define MINI_AIRCRAFT_ROTOR_HPP_

#include <iostream>
#include <vector>

#include "mini/geometry/pi.hpp"
#include "mini/aircraft/blade.hpp"

namespace mini {
namespace aircraft {

template <typename Scalar>
class Rotor {
 public:
  using Blade = mini::aircraft::Blade<Scalar>;
  using Frame = typename Blade::Frame;
  using Vector = typename Blade::Vector;

 protected:
  std::vector<Blade> blades_;
  Frame frame_;
  Vector origin_;
  Scalar initial_azimuth_{0}/* deg */;
  Scalar azimuth_/* deg */, omega_/* deg / s */;

 public:
  /**
   * @brief Set the rotating speed of this `Rotor`.
   * 
   * @param rps The rotating speed in revolutions per second.
   * @return Rotor& The reference to this `Rotor`.
   */
  Rotor& SetRevolutionsPerSecond(Scalar rps) {
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
  Rotor& SetInitialAzimuth(Scalar deg) {
    initial_azimuth_ = deg;
    SetAzimuth(deg);
    return *this;
  }
  Rotor& UpdateAzimuth(Scalar sec) {
    auto deg = initial_azimuth_;
    deg += sec * GetDegreesPerSecond();
    SetAzimuth(deg);
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
  Scalar GetDegreesPerSecond() const {
    return omega_;
  }
  /**
   * @brief Get the rotating speed of this `Rotor`.
   * 
   * @return Scalar The rotating speed in radians per second.
   */
  Scalar GetRadiansPerSecond() const {
    return mini::geometry::deg2rad(omega_);
  }

  /**
   * @brief Get the azimuth of this `Rotor`.
   * 
   * @return Scalar The azimuth in degrees.
   */
  Scalar GetAzimuth() const {
    return azimuth_;
  }

  const Vector& GetOrigin() const {
    return origin_;
  }

  const Frame& GetFrame() const {
    return frame_;
  }

  const Blade& GetBlade(int i) const {
    return blades_.at(i);
  }
};

}  // namespace aircraft
}  // namespace mini

#endif  // MINI_AIRCRAFT_ROTOR_HPP_
