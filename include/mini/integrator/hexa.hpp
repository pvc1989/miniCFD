//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_INTEGRATOR_HEXA_HPP_
#define MINI_INTEGRATOR_HEXA_HPP_

#include <Eigen/Dense>

#include <type_traits>

#include "mini/integrator/line.hpp"
namespace mini {
namespace integrator {
/**
 * @brief 
 * 
 * @tparam Scalar 
 * @tparam Qx 
 * @tparam Qy 
 * @tparam Qz 
 */
template <typename Scalar = double, int Qx = 4, int Qy = 4, int Qz = 4>
class Hexa {
  using Mat3x3 = Eigen::Matrix<Scalar, 3, 3>;
  using Mat1x8 = Eigen::Matrix<Scalar, 1, 8>;
  using Mat8x1 = Eigen::Matrix<Scalar, 8, 1>;
  using Mat3x8 = Eigen::Matrix<Scalar, 3, 8>;
  using Mat8x3 = Eigen::Matrix<Scalar, 8, 3>;
  using Mat3x1 = Eigen::Matrix<Scalar, 3, 1>;
  using Mat10x1 = Eigen::Matrix<Scalar, 10, 1>;

  using Arr1x8 = Eigen::Array<Scalar, 1, 8>;
  using Arr8x1 = Eigen::Array<Scalar, 8, 1>;
  using Arr3x8 = Eigen::Array<Scalar, 3, 8>;
  using Arr8x3 = Eigen::Array<Scalar, 8, 3>;

  using GaussX = GaussIntegrator<Scalar, Qx>;
  using GaussY = GaussIntegrator<Scalar, Qy>;
  using GaussZ = GaussIntegrator<Scalar, Qz>;

 public:
  using Real = Scalar;
  using LocalCoord = Eigen::Matrix<Scalar, 3, 1>;
  using GlobalCoord = Eigen::Matrix<Scalar, 3, 1>;
 private:
  static const Arr1x8 x_local_i_;
  static const Arr1x8 y_local_i_;
  static const Arr1x8 z_local_i_;
  Mat3x8 xyz_global_3x8_;
  static const std::array<LocalCoord, Qx * Qy * Qz> points_;
  static const std::array<Scalar, Qx * Qy * Qz> weights_;

 public:
  static constexpr int CountQuadPoints() {
    return Qx * Qy * Qz;
  }
  static constexpr auto BuildPoints() {
    std::array<LocalCoord, Qx * Qy * Qz> points;
    int n = 0;
    for (int i = 0; i < Qx; ++i) {
      for (int j = 0; j < Qy; ++j) {
        for (int k = 0; k < Qz; ++k) {
          points[n][0] = GaussX::points[i];
          points[n][1] = GaussY::points[j];
          points[n][2] = GaussZ::points[k];
          n++;
        }
      }
    }
    return points;
  }
  static constexpr auto BuildWeights() {
    std::array<Scalar, Qx * Qy * Qz> weights;
    int n = 0;
    for (int i = 0; i < Qx; ++i) {
      for (int j = 0; j < Qy; ++j) {
        for (int k = 0; k < Qz; ++k) {
          weights[n++] = GaussX::weights[i] * GaussY::weights[j]
              * GaussZ::weights[k];
        }
      }
    }
    return weights;
  }

 private:
  static Mat8x1 shape_8x1(Mat3x1 xyz_local) {
    Arr1x8 n_1x8;
    n_1x8  = (1 + x_local_i_ * xyz_local[0]);
    n_1x8 *= (1 + y_local_i_ * xyz_local[1]);
    n_1x8 *= (1 + z_local_i_ * xyz_local[2]);
    n_1x8 /= 8.0;
    return n_1x8.transpose();
  }
  static Mat8x1 shape_8x1(Scalar x_local, Scalar y_local, Scalar z_local) {
    Arr1x8 n_1x8;
    n_1x8  = (1 + x_local_i_ * x_local);
    n_1x8 *= (1 + y_local_i_ * y_local);
    n_1x8 *= (1 + z_local_i_ * z_local);
    n_1x8 /= 8.0;
    return n_1x8.transpose();
  }
  static Mat8x3 diff_shape_local_8x3(
      Scalar x_local, Scalar y_local, Scalar z_local) {
    Arr8x3 dn;
    Arr8x1 factor_x = x_local_i_.transpose() * x_local; factor_x += 1;
    Arr8x1 factor_y = y_local_i_.transpose() * y_local; factor_y += 1;
    Arr8x1 factor_z = z_local_i_.transpose() * z_local; factor_z += 1;
    dn.col(0) << x_local_i_.transpose() * factor_y * factor_z;
    dn.col(1) << y_local_i_.transpose() * factor_x * factor_z;
    dn.col(2) << z_local_i_.transpose() * factor_x * factor_y;
    dn /= 8.0;
    return dn;
  }
  Mat3x3 jacobian(Scalar x_local, Scalar y_local, Scalar z_local) const {
    return xyz_global_3x8_ * diff_shape_local_8x3(x_local, y_local, z_local);
  }
  template <typename Callable, typename MatJ>
  static Mat3x1 root(
      Callable&& func, Mat3x1 x, MatJ&& matj, Scalar xtol = 1e-5) {
    Mat3x1 res;
    do {
      res = matj(x).partialPivLu().solve(func(x));
      x -= res;
    } while (res.norm() > xtol);
    return x;
  }

 public:
  static constexpr int CellDim() {
    return 3;
  }
  static constexpr int PhysDim() {
    return 3;
  }
  static LocalCoord const& GetCoord(int i) {
    return points_[i];
  }
  static Scalar const& GetWeight(int i) {
    return weights_[i];
  }
public:
  Hexa(Mat3x8 const& xyz_global) {
    xyz_global_3x8_ = xyz_global;
  }
  GlobalCoord local_to_global_Dx1(
      Scalar x_local, Scalar y_local, Scalar z_local) const {
    return xyz_global_3x8_ * shape_8x1(x_local, y_local, z_local);
  }
  GlobalCoord local_to_global_Dx1(LocalCoord const& xyz_local) const {
    return xyz_global_3x8_ * shape_8x1(xyz_local);
  }
  Mat3x3 jacobian(const LocalCoord& xyz_local) const {
    return jacobian(xyz_local[0], xyz_local[1], xyz_local[2]);
  }
  GlobalCoord global_to_local_3x1(
      Scalar x_global, Scalar y_global, Scalar z_global) const {
    Mat3x1 xyz_global = {x_global, y_global, z_global};
    auto func = [this, &xyz_global](Mat3x1 const& xyz_local) {
      auto res = local_to_global_Dx1(xyz_local);
      return res -= xyz_global;
    };
    auto jac = [this](LocalCoord const& xyz_local) {
      return jacobian(xyz_local);
    };
    Mat3x1 xyz0 = {0, 0, 0};
    return root(func, xyz0, jac);
  }
  GlobalCoord global_to_local_3x1(LocalCoord xyz_global) const {
    return global_to_local_3x1(xyz_global[0], xyz_global[1], xyz_global[2]);
  }
};
template <typename Scalar, int Qx, int Qy, int Qz>
typename Hexa<Scalar, Qx, Qy, Qz>::Arr1x8 const
Hexa<Scalar, Qx, Qy, Qz>::x_local_i_ = {-1, +1, +1, -1, -1, +1, +1, -1};

template <typename Scalar, int Qx, int Qy, int Qz>
typename Hexa<Scalar, Qx, Qy, Qz>::Arr1x8 const
Hexa<Scalar, Qx, Qy, Qz>::y_local_i_ = {-1, -1, +1, +1, -1, -1, +1, +1};

template <typename Scalar, int Qx, int Qy, int Qz>
typename Hexa<Scalar, Qx, Qy, Qz>::Arr1x8 const
Hexa<Scalar, Qx, Qy, Qz>::z_local_i_ = {-1, -1, -1, -1, +1, +1, +1, +1};

template <typename Scalar, int Qx, int Qy, int Qz>
std::array<typename Hexa<Scalar, Qx, Qy, Qz>::LocalCoord, Qx * Qy * Qz> const
Hexa<Scalar, Qx, Qy, Qz>::points_ = Hexa<Scalar, Qx, Qy, Qz>::BuildPoints();

template <typename Scalar, int Qx, int Qy, int Qz>
std::array<Scalar, Qx * Qy * Qz> const
Hexa<Scalar, Qx, Qy, Qz>::weights_ = Hexa<Scalar, Qx, Qy, Qz>::BuildWeights();

}  // namespace integrator
}  // namespace mini

#endif  // MINI_INTEGRATOR_HEXA_HPP_
