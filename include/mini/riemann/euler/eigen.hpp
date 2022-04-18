// Copyright 2021 PEI WeiCheng and JIANG YuYan
#ifndef MINI_RIEMANN_EULER_EIGEN_HPP_
#define MINI_RIEMANN_EULER_EIGEN_HPP_

#include "mini/algebra/eigen.hpp"
#include "mini/riemann/euler/types.hpp"

namespace mini {
namespace riemann {
namespace euler {

template <typename IdealGas>
class EigenMatrices {
  using Scalar = typename IdealGas::Scalar;
  using Conservative = Conservatives<Scalar, 3>;
  using Primitive = Primitives<Scalar, 3>;

 public:
  using Mat5x5 = algebra::Matrix<Scalar, 5, 5>;
  using Mat5x1 = algebra::Matrix<Scalar, 5, 1>;
  using Mat3x1 = algebra::Matrix<Scalar, 3, 1>;
  using Mat3x3 = algebra::Matrix<Scalar, 3, 3>;

  Mat5x5 L, R;

  EigenMatrices() = default;
  EigenMatrices(const Primitive& primitive,  // orthonormal vectors:
      const Mat3x1& nu, const Mat3x1& mu, const Mat3x1& pi) {
    constexpr int x = 0, y = 1, z = 2;
    auto u_x = primitive.u(), u_y = primitive.v(), u_z = primitive.w();
    auto u_nu = (u_x * nu[x] + u_y * nu[y] + u_z * nu[z]);
    auto u_mu = (u_x * mu[x] + u_y * mu[y] + u_z * mu[z]);
    auto u_pi = (u_x * pi[x] + u_y * pi[y] + u_z * pi[z]);
    auto a = IdealGas::GetSpeedOfSound(primitive);
    auto& velocity = primitive.momentum();
    auto ek = velocity.dot(velocity) / 2;
    auto e0 = a * a * IdealGas::OneOverGammaMinusOne();
    auto h0 = e0 + ek;
    // build Mat5x5 R
    R.col(0) << 1, u_x - a*nu[x], u_y - a*nu[y], u_z - a*nu[z], h0 - a*u_nu;
    R.col(1) << 1, u_x, u_y, u_z, ek;
    R.col(2) << 0, mu[x], mu[y], mu[z], u_mu;
    R.col(3) << 0, pi[x], pi[y], pi[z], u_pi;
    R.col(4) << 1, u_x + a*nu[x], u_y + a*nu[y], u_z + a*nu[z], h0 + a*u_nu;
    // build Mat5x5 L
    auto b1 = 1 / e0, b2 = b1 * ek;
    L.col(0) << (b2 + u_nu / a) / 2, 1 - b2, -u_mu, -u_pi,
                (b2 - u_nu / a) / 2;
    L.col(1) << -(b1 * u_x + nu[x] / a) / 2, b1 * u_x, mu[x], pi[x],
                -(b1 * u_x - nu[x] / a) / 2;
    L.col(2) << -(b1 * u_y + nu[y] / a) / 2, b1 * u_y, mu[y], pi[y],
                -(b1 * u_y - nu[y] / a) / 2;
    L.col(3) << -(b1 * u_z + nu[z] / a) / 2, b1 * u_z, mu[z], pi[z],
                -(b1 * u_z - nu[z] / a) / 2;
    L.col(4) << b1 / 2, -b1, 0, 0, b1 / 2;
  }
  EigenMatrices(const Conservative &conservative,  // orthonormal vectors:
      const Mat3x1& nu, const Mat3x1& mu, const Mat3x1& pi) {
    auto primitive = IdealGas::ConservativeToPrimitive(conservative);
    *this = EigenMatrices(primitive, nu, mu, pi);
  }
  EigenMatrices(const Mat5x1 &tuple, const Mat3x3& frame) {
    const auto &consv = *reinterpret_cast<const Conservative *>(&tuple);
    *this = EigenMatrices(consv, frame.col(0), frame.col(1), frame.col(2));
  }
};

}  // namespace euler
}  // namespace riemann
}  // namespace mini

#endif  //  MINI_RIEMANN_EULER_EIGEN_HPP_
