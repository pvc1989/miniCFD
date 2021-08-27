//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_INTEGRATOR_QUAD_HPP_
#define MINI_INTEGRATOR_QUAD_HPP_

#include "mini/algebra/eigen.hpp"

#include "mini/integrator/line.hpp"
#include "mini/integrator/face.hpp"

namespace mini {
namespace integrator {

/**
 * @brief 
 * 
 * @tparam Scalar 
 * @tparam D the dim of the physical space
 * @tparam Qx 
 * @tparam Qy 
 */
template <typename Scalar = double, int D = 2, int Qx = 4, int Qy = 4>
class Quad : public Face<Scalar, D> {
  using Arr1x4 = algebra::Array<Scalar, 1, 4>;
  using Arr4x1 = algebra::Array<Scalar, 4, 1>;
  using Arr4x2 = algebra::Array<Scalar, 4, 2>;

  using MatDx4 = algebra::Matrix<Scalar, D, 4>;
  using MatDx2 = algebra::Matrix<Scalar, D, 2>;
  using MatDx1 = algebra::Matrix<Scalar, D, 1>;
  using Mat4x1 = algebra::Matrix<Scalar, 4, 1>;
  using Mat4x2 = algebra::Matrix<Scalar, 4, 2>;
  using Mat2x1 = algebra::Matrix<Scalar, 2, 1>;

  using GaussX = GaussIntegrator<Scalar, Qx>;
  using GaussY = GaussIntegrator<Scalar, Qy>;

 public:
  using Real = Scalar;
  using LocalCoord = algebra::Matrix<Scalar, 2, 1>;
  using GlobalCoord = algebra::Matrix<Scalar, D, 1>;
  static const Arr1x4 x_local_i_;
  static const Arr1x4 y_local_i_;
  MatDx4 xyz_global_Dx4_;
  static const std::array<LocalCoord, Qx * Qy> points_;
  static const std::array<Scalar, Qx * Qy> weights_;

  int CountQuadPoints() const override {
    return Qx * Qy;
  }
  static constexpr auto BuildPoints() {
    std::array<LocalCoord, Qx * Qy> points;
    int k = 0;
    for (int i = 0; i < Qx; ++i) {
      for (int j = 0; j < Qy; ++j) {
        points[k][0] = GaussX::points[i];
        points[k][1] = GaussY::points[j];
        k++;
      }
    }
    return points;
  }
  static constexpr auto BuildWeights() {
    std::array<Scalar, Qx * Qy> weights;
    int k = 0;
    for (int i = 0; i < Qx; ++i) {
      for (int j = 0; j < Qy; ++j) {
        weights[k++] = GaussX::weights[i] * GaussY::weights[j];
      }
    }
    return weights;
  }

 private:
  static Mat4x1 shape_4x1(Mat2x1 xy_local) {
    Arr1x4 n_1x4;
    n_1x4  = (1 + x_local_i_ * xy_local[0]);
    n_1x4 *= (1 + y_local_i_ * xy_local[1]);
    n_1x4 /= 4.0;
    return n_1x4.transpose();
  }
  static Mat4x1 shape_4x1(Scalar x, Scalar y) {
    Arr1x4 n_1x4;
    n_1x4  = (1 + x_local_i_ * x);
    n_1x4 *= (1 + y_local_i_ * y);
    n_1x4 /= 4.0;
    return n_1x4.transpose();
  }
  static Mat4x2 diff_shape_local_4x2(Scalar x_local, Scalar y_local) {
    Arr4x2 dn;
    Arr4x1 factor_x = x_local_i_.transpose() * x_local; factor_x += 1;
    Arr4x1 factor_y = y_local_i_.transpose() * y_local; factor_y += 1;
    dn.col(0) << x_local_i_.transpose() * factor_y;
    dn.col(1) << y_local_i_.transpose() * factor_x;
    dn /= 4.0;
    return dn;
  }
  MatDx2 jacobian(Scalar x_local, Scalar y_local) const {
    return xyz_global_Dx4_ * diff_shape_local_4x2(x_local, y_local);
  }

 public:
  LocalCoord const& GetCoord(int i) const override {
    return points_[i];
  }
  Scalar const& GetWeight(int i) const override {
    return weights_[i];
  }
  MatDx1 GetCenter() const override {
    MatDx1 c = xyz_global_Dx4_.col(0);
    for (int i = 1; i < 4; ++i)
      c += xyz_global_Dx4_.col(i);
    c /= 4;
    return c;
  }

 public:
  explicit Quad(MatDx4 const& xyz_global) {
    xyz_global_Dx4_ = xyz_global;
  }
  Quad(MatDx1 const& p0, MatDx1 const& p1, MatDx1 const& p2, MatDx1 const& p3) {
    xyz_global_Dx4_.col(0) = p0; xyz_global_Dx4_.col(1) = p1;
    xyz_global_Dx4_.col(2) = p2; xyz_global_Dx4_.col(3) = p3;
  }
  Quad(std::initializer_list<MatDx1> il) {
    assert(il.size() == 4);
    auto p = il.begin();
    for (int i = 0; i < 4; ++i) {
      xyz_global_Dx4_[i] = p[i];
    }
  }
  MatDx1 local_to_global_Dx1(const Mat2x1& xy_local) const override {
    return xyz_global_Dx4_ * shape_4x1(xy_local);
  }
  MatDx1 local_to_global_Dx1(Scalar x, Scalar y) const {
    return xyz_global_Dx4_ * shape_4x1(x, y);
  }
  MatDx2 jacobian(const LocalCoord& xy_local) const override {
    return jacobian(xy_local[0], xy_local[1]);
  }
};

template <typename Scalar, int D, int Qx, int Qy>
typename Quad<Scalar, D, Qx, Qy>::Arr1x4 const
Quad<Scalar, D, Qx, Qy>::x_local_i_ = {-1, +1, +1, -1};

template <typename Scalar, int D, int Qx, int Qy>
typename Quad<Scalar, D, Qx, Qy>::Arr1x4 const
Quad<Scalar, D, Qx, Qy>::y_local_i_ = {-1, -1, +1, +1};

template <typename Scalar, int D, int Qx, int Qy>
std::array<typename Quad<Scalar, D, Qx, Qy>::LocalCoord, Qx * Qy> const
Quad<Scalar, D, Qx, Qy>::points_ = Quad<Scalar, D, Qx, Qy>::BuildPoints();

template <typename Scalar, int D, int Qx, int Qy>
std::array<Scalar, Qx * Qy> const
Quad<Scalar, D, Qx, Qy>::weights_ = Quad<Scalar, D, Qx, Qy>::BuildWeights();

}  // namespace integrator
}  // namespace mini

#endif  // MINI_INTEGRATOR_QUAD_HPP_
