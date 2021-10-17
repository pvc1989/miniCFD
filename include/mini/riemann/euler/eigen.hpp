// Copyright 2021 PEI WeiCheng and JIANG YuYan
#ifndef MINI_RIEMANN_EULER_EIGEN_HPP_
#define MINI_RIEMANN_EULER_EIGEN_HPP_

#include <cmath>
#include <initializer_list>

#include "mini/algebra/eigen.hpp"
#include "mini/riemann/euler/types.hpp"

namespace mini {
namespace riemann {
namespace euler {

template <typename Scalar, typename IdealGas>
class EigenMatrices {
 public:
  using Mat5x5 = algebra::Matrix<Scalar, 5, 5>;
  using Mat3x1 = algebra::Matrix<Scalar, 3, 1>;

  Mat5x5 L, R;

  EigenMatrices(Primitive<3>& primitive, // orthonormal vectors:
      const Mat3x1& nu, const Mat3x1& sigma, const Mat3x1& pi) {
    constexpr int x = 0, y = 1, z = 2;
    auto u_x = primitive.u(), u_y = primitive.v(), u_z = primitive.w();
    auto u_nu = (u_x * nu[x] + u_y * nu[y] + u_z * nu[z]);
    auto u_sigma = (u_x * sigma[x] + u_y * sigma[y] + u_z * sigma[z]);
    auto u_pi = (u_x * pi[x] + u_y * pi[y] + u_z * pi[z]);
    auto a = IdealGas::GetSpeedOfSound(primitive);
    auto& velocity = primitive.momentum;
    auto ek = velocity.Dot(velocity) / 2;
    auto e0 = a * a * IdealGas::OneOverGammaMinusOne();
    auto h0 = e0 + ek;
    // build Mat5x5 R
    R.col(0) << 1, u_x - a*nu[x], u_y - a*nu[y], u_z - a*nu[z], h0 - a*u_nu;
    R.col(1) << 1, u_x, u_y, u_z, ek;
    R.col(2) << 0, sigma[x], sigma[y], sigma[z], u_sigma;
    R.col(3) << 0, pi[x], pi[y], pi[z], u_pi;
    R.col(4) << 1, u_x + a*nu[x], u_y + a*nu[y], u_z + a*nu[z], h0 + a*u_nu;
    // build Mat5x5 L
    auto b1 = 1 / e0, b2 = b1 * ek;
    L.col(0) << (b2 + u_nu / a) / 2, 1 - b2, -u_sigma, -u_pi,
                (b2 - u_nu / a) / 2;
    L.col(1) << -(b1 * u_x + nu[x] / a) / 2, b1 * u_x, sigma[x], pi[x],
                -(b1 * u_x - nu[x] / a) / 2;
    L.col(2) << -(b1 * u_y + nu[y] / a) / 2, b1 * u_y, sigma[y], pi[y],
                -(b1 * u_y - nu[y] / a) / 2;
    L.col(3) << -(b1 * u_z + nu[z] / a) / 2, b1 * u_z, sigma[z], pi[z],
                -(b1 * u_z - nu[z] / a) / 2;
    L.col(4) << b1 / 2, -b1, 0, 0, b1 / 2;
  }
};

}  // namespace euler
}  // namespace riemann
}  // namespace mini

#endif  //  MINI_RIEMANN_EULER_EIGEN_HPP_
