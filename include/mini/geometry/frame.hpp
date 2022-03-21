// Copyright 2022 PEI Weicheng
#ifndef MINI_GEOMETRY_FRAME_HPP_
#define MINI_GEOMETRY_FRAME_HPP_

#include <utility>

#include "mini/algebra/eigen.hpp"

namespace mini {
namespace geometry {

template <typename Scalar>
class Frame {
 public:
  using Vector = mini::algebra::Matrix<Scalar, 3, 1>;
  const Vector& X() const {
    return x_;
  }
  const Vector& Y() const {
    return y_;
  }
  const Vector& Z() const {
    return z_;
  }

 public:
  static Scalar pi() {
    return 3.1415926535897932384626433832795028841971693993751;
  }
  Frame& RotateX(Scalar angle) {
    auto [cos, sin] = CosSin(angle);
    Vector new_y = Y() * cos + Z() * sin;
    Vector new_z = Z() * cos - Y() * sin;
    y_ = new_y;
    z_ = new_z;
    return *this;
  }
  Frame& RotateY(Scalar angle) {
    auto [cos, sin] = CosSin(angle);
    Vector new_x = X() * cos - Z() * sin;
    Vector new_z = Z() * cos + X() * sin;
    x_ = new_x;
    z_ = new_z;
    return *this;
  }
  Frame& RotateZ(Scalar angle) {
    auto [cos, sin] = CosSin(angle);
    Vector new_x = X() * cos + Y() * sin;
    Vector new_y = Y() * cos - X() * sin;
    x_ = new_x;
    y_ = new_y;
    return *this;
  }

 private:
  Vector x_{1, 0, 0}, y_{0, 1, 0}, z_{0, 0, 1};

  static std::pair<Scalar, Scalar> CosSin(Scalar angle) {
    return { std::cos(angle), std::sin(angle) };
  }
};

}  // namespace geometry
}  // namespace mini

#endif  // MINI_GEOMETRY_FRAME_HPP_
