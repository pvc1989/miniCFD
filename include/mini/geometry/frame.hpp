// Copyright 2022 PEI Weicheng
#ifndef MINI_GEOMETRY_FRAME_HPP_
#define MINI_GEOMETRY_FRAME_HPP_

#include <concepts>
#include <iostream>
#include <utility>

#include "mini/algebra/eigen.hpp"
#include "mini/geometry/pi.hpp"

namespace mini {
namespace geometry {

template <std::floating_point Scalar>
class Frame {
 public:
  using Vector = mini::algebra::Matrix<Scalar, 3, 1>;
  const Vector &X() const {
    return x_;
  }
  const Vector &Y() const {
    return y_;
  }
  const Vector &Z() const {
    return z_;
  }

 public:
  Frame &RotateX(Scalar deg) {
    auto [cos, sin] = CosSin(deg);
    Vector new_y = Y() * cos + Z() * sin;
    Vector new_z = Z() * cos - Y() * sin;
    y_ = new_y;
    z_ = new_z;
    return *this;
  }
  Frame &RotateY(Scalar deg) {
    auto [cos, sin] = CosSin(deg);
    Vector new_x = X() * cos - Z() * sin;
    Vector new_z = Z() * cos + X() * sin;
    x_ = new_x;
    z_ = new_z;
    return *this;
  }
  Frame &RotateZ(Scalar deg) {
    auto [cos, sin] = CosSin(deg);
    Vector new_x = X() * cos + Y() * sin;
    Vector new_y = Y() * cos - X() * sin;
    x_ = new_x;
    y_ = new_y;
    return *this;
  }

 private:
  Vector x_{1, 0, 0}, y_{0, 1, 0}, z_{0, 0, 1};
};

template <std::floating_point Scalar>
std::ostream &operator<<(std::ostream &out, const Frame<Scalar> &frame) {
  out << "[" << frame.X().transpose() << "] ";
  out << "[" << frame.Y().transpose() << "] ";
  out << "[" << frame.Z().transpose() << "] ";
  return out;
}

}  // namespace geometry
}  // namespace mini

#endif  // MINI_GEOMETRY_FRAME_HPP_
