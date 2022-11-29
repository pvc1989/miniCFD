//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_POLYNOMIAL_BASIS_HPP_
#define MINI_POLYNOMIAL_BASIS_HPP_

#include <cmath>
#include <iostream>
#include <type_traits>

#include "mini/algebra/eigen.hpp"

#include "mini/integrator/function.hpp"
#include "mini/integrator/face.hpp"
#include "mini/integrator/cell.hpp"

namespace mini {
namespace polynomial {

template <typename Scalar, int kDimensions, int kDegrees>
class Raw;

template <typename Scalar>
class Raw<Scalar, 2, 1> {
 public:
  static constexpr int N = 3;  // the number of components
  using MatNx1 = algebra::Matrix<Scalar, N, 1>;
  using Coord = algebra::Matrix<Scalar, 2, 1>;

  static MatNx1 GetValue(const Coord &xy) {
    MatNx1 col = { 1, xy[0], xy[1] };
    return col;
  }
};

template <typename Scalar>
class Raw<Scalar, 2, 2> {
 public:
  static constexpr int N = 6;  // the number of components
  using MatNx1 = algebra::Matrix<Scalar, N, 1>;
  using Coord = algebra::Matrix<Scalar, 2, 1>;

  static MatNx1 GetValue(const Coord &xy) {
    auto x = xy[0], y = xy[1];
    MatNx1 col = { 1, x, y, x * x, x * y, y * y };
    return col;
  }
};

template <typename Scalar>
class Raw<Scalar, 2, 3> {
 public:
  static constexpr int N = 10;  // the number of components
  using MatNx1 = algebra::Matrix<Scalar, N, 1>;
  using Coord = algebra::Matrix<Scalar, 2, 1>;

  static MatNx1 GetValue(const Coord &xy) {
    auto x = xy[0], y = xy[1];
    auto x_x = x * x, x_y = x * y, y_y = y * y;
    MatNx1 col = { 1, x, y, x_x, x_y, y_y,
        x_x * x, x_x * y, x * y_y, y * y_y };
    return col;
  }
};

template <typename Scalar>
class Raw<Scalar, 3, 0> {
 public:
  static constexpr int N = 1;  // the number of components
  using MatNx1 = algebra::Matrix<Scalar, N, 1>;
  using Coord = algebra::Matrix<Scalar, 3, 1>;

  static MatNx1 GetValue(const Coord &xyz) {
    MatNx1 col; col(0, 0) = 1;
    return col;
  }

  template <typename MatKxN>
  static MatKxN GetPdvValue(const Coord &xyz, const MatKxN &coeff) {
    MatKxN res; res.setZero();
    return res;
  }

  template <int K>
  static auto GetGradValue(const Coord &xyz,
      const algebra::Matrix<Scalar, K, N> &coeff) {
    algebra::Matrix<Scalar, K, 3> res; res.setZero();
    return res;
  }

  template <int K>
  static auto GetSmoothness(
      const algebra::Matrix<Scalar, K, N> &integral, Scalar volume) {
    using MatKx1 = algebra::Matrix<Scalar, K, 1>;
    MatKx1 smoothness; smoothness.setZero();
    return smoothness;
  }
};

template <typename Scalar>
class Raw<Scalar, 3, 1> {
 public:
  static constexpr int N = 4;  // the number of components
  using MatNx1 = algebra::Matrix<Scalar, N, 1>;
  using MatNx3 = algebra::Matrix<Scalar, N, 3>;
  using Coord = algebra::Matrix<Scalar, 3, 1>;

  static MatNx1 GetValue(const Coord &xyz) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    MatNx1 col = { 1, x, y, z };
    return col;
  }
  template <typename MatKxN>
  static MatKxN GetPdvValue(const Coord &xyz, const MatKxN &coeff) {
    MatKxN res = coeff; res.col(0).setZero();
    return res;
  }

  template <int K>
  static auto GetGradValue(const Coord &xyz,
      const algebra::Matrix<Scalar, K, N> &coeff) {
    algebra::Matrix<Scalar, K, 3> res;
    // pdv_x
    res.col(0) = coeff.col(1);
    // pdv_y
    res.col(1) = coeff.col(2);
    // pdv_z
    res.col(2) = coeff.col(3);
    return res;
  }

  template <int K>
  static auto GetSmoothness(
      const algebra::Matrix<Scalar, K, N> &integral, Scalar volume) {
    using MatKx1 = algebra::Matrix<Scalar, K, 1>;
    MatKx1 smoothness = integral.col(1);
    smoothness += integral.col(2);
    smoothness += integral.col(3);
    return smoothness;
  }
};

template <typename Scalar>
class Raw<Scalar, 3, 2> {
 public:
  static constexpr int N = 10;  // the number of components
  using MatNx1 = algebra::Matrix<Scalar, N, 1>;
  using MatNx3 = algebra::Matrix<Scalar, N, 3>;
  using Coord = algebra::Matrix<Scalar, 3, 1>;

  static MatNx1 GetValue(const Coord &xyz) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    MatNx1 col = { 1, x, y, z, x * x, x * y, x * z, y * y, y * z, z * z };
    return col;
  }
  template <typename MatKxN>
  static MatKxN GetPdvValue(const Coord &xyz, const MatKxN &coeff) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    MatKxN res = coeff; res.col(0).setZero();
    // pdv_x
    assert(res.col(1) == coeff.col(1));
    res.col(1) += coeff.col(4) * (2 * x);
    res.col(1) += coeff.col(5) * y;
    res.col(1) += coeff.col(6) * z;
    // pdv_y
    assert(res.col(2) == coeff.col(2));
    res.col(2) += coeff.col(5) * x;
    res.col(2) += coeff.col(7) * (2 * y);
    res.col(2) += coeff.col(8) * z;
    // pdv_z
    assert(res.col(3) == coeff.col(3));
    res.col(3) += coeff.col(6) * x;
    res.col(3) += coeff.col(8) * y;
    res.col(3) += coeff.col(9) * (2 * z);
    // pdv_xx
    res.col(4) += coeff.col(4);
    assert(res.col(4) == coeff.col(4) * 2);
    // pdv_xy
    assert(res.col(5) == coeff.col(5));
    // pdv_xz
    assert(res.col(6) == coeff.col(6));
    // pdv_yy
    res.col(7) += coeff.col(7);
    assert(res.col(7) == coeff.col(7) * 2);
    // pdv_yz
    assert(res.col(8) == coeff.col(8));
    // pdv_zz
    res.col(9) += coeff.col(9);
    assert(res.col(9) == coeff.col(9) * 2);
    return res;
  }

  template <int K>
  static auto GetGradValue(const Coord &xyz,
      const algebra::Matrix<Scalar, K, N> &coeff) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    algebra::Matrix<Scalar, K, 3> res;
    // pdv_x
    res.col(0) = coeff.col(1);
    res.col(0) += coeff.col(4) * (2 * x);
    res.col(0) += coeff.col(5) * y;
    res.col(0) += coeff.col(6) * z;
    // pdv_y
    res.col(1) = coeff.col(2);
    res.col(1) += coeff.col(5) * x;
    res.col(1) += coeff.col(7) * (2 * y);
    res.col(1) += coeff.col(8) * z;
    // pdv_z
    res.col(2) = coeff.col(3);
    res.col(2) += coeff.col(6) * x;
    res.col(2) += coeff.col(8) * y;
    res.col(2) += coeff.col(9) * (2 * z);
    return res;
  }

  template <int K>
  static auto GetSmoothness(
      const algebra::Matrix<Scalar, K, N> &integral, Scalar volume) {
    using MatKx1 = algebra::Matrix<Scalar, K, 1>;
    auto w1  // weight of 1st-order partial derivatives
        = std::pow(volume, 2./3-1);
    MatKx1 smoothness = integral.col(1);
    smoothness += integral.col(2);
    smoothness += integral.col(3);
    smoothness *= w1;
    auto w2  // weight of 2nd-order partial derivatives
        = std::pow(volume, 4./3-1);
    smoothness += integral.col(4) * w2;
    smoothness += integral.col(5) * w2;
    smoothness += integral.col(6) * w2;
    smoothness += integral.col(7) * w2;
    smoothness += integral.col(8) * w2;
    smoothness += integral.col(9) * w2;
    return smoothness;
  }
};

template <typename Scalar>
class Raw<Scalar, 3, 3> {
  static constexpr int X{1}, Y{2}, Z{3};
  static constexpr int XX{4}, XY{5}, XZ{6}, YY{7}, YZ{8}, ZZ{9};
  static constexpr int XXX{10}, XXY{11}, XXZ{12}, XYY{13}, XYZ{14}, XZZ{15};
  static constexpr int YYY{16}, YYZ{17}, YZZ{18}, ZZZ{19};

 public:
  static constexpr int N = 20;  // the number of components
  using MatNx1 = algebra::Matrix<Scalar, N, 1>;
  using MatNx3 = algebra::Matrix<Scalar, N, 3>;
  using Coord = algebra::Matrix<Scalar, 3, 1>;

  static MatNx1 GetValue(const Coord &xyz) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    auto xx{x * x}, xy{x * y}, xz{x * z}, yy{y * y}, yz{y * z}, zz{z * z};
    MatNx1 col = { 1, x, y, z, xx, xy, xz, yy, yz, zz,
        x * xx, x * xy, x * xz, x * yy, x * yz, x * zz,
        y * yy, y * yz, y * zz, z * zz };
    return col;
  }
  template <typename MatKxN>
  static MatKxN GetPdvValue(const Coord &xyz, const MatKxN &coeff) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    auto xx{x * x}, xy{x * y}, xz{x * z}, yy{y * y}, yz{y * z}, zz{z * z};
    MatKxN res = coeff; res.col(0).setZero();
    // pdv_x
    assert(res.col(X) == coeff.col(X));
    res.col(X) += coeff.col(XX) * (2 * x);
    res.col(X) += coeff.col(XY) * y;
    res.col(X) += coeff.col(XZ) * z;
    res.col(X) += coeff.col(XXX) * (3 * xx);
    res.col(X) += coeff.col(XXY) * (2 * xy);
    res.col(X) += coeff.col(XXZ) * (2 * xz);
    res.col(X) += coeff.col(XYY) * yy;
    res.col(X) += coeff.col(XYZ) * yz;
    res.col(X) += coeff.col(XZZ) * zz;
    // pdv_y
    assert(res.col(Y) == coeff.col(Y));
    res.col(Y) += coeff.col(XY) * x;
    res.col(Y) += coeff.col(YY) * (2 * y);
    res.col(Y) += coeff.col(YZ) * z;
    res.col(Y) += coeff.col(XXY) * xx;
    res.col(Y) += coeff.col(XYZ) * xz;
    res.col(Y) += coeff.col(XYY) * (2 * xy);
    res.col(Y) += coeff.col(YYY) * (3 * yy);
    res.col(Y) += coeff.col(YYZ) * (2 * yz);
    res.col(Y) += coeff.col(YZZ) * zz;
    // pdv_z
    assert(res.col(Z) == coeff.col(Z));
    res.col(Z) += coeff.col(XZ) * x;
    res.col(Z) += coeff.col(YZ) * y;
    res.col(Z) += coeff.col(ZZ) * (2 * z);
    res.col(Z) += coeff.col(XXZ) * xx;
    res.col(Z) += coeff.col(XYZ) * xy;
    res.col(Z) += coeff.col(XZZ) * (2 * xz);
    res.col(Z) += coeff.col(YYZ) * yy;
    res.col(Z) += coeff.col(YZZ) * (2 * yz);
    res.col(Z) += coeff.col(ZZZ) * (3 * zz);
    // pdv_xx
    res.col(XX) += coeff.col(XXY) * y;
    res.col(XX) += coeff.col(XXZ) * z;
    res.col(XX) += coeff.col(XXX) * x * 3;
    res.col(XX) += res.col(XX);
    assert(res.col(XX) == 2 * (coeff.col(XX) + coeff.col(XXY) * y
        + coeff.col(XXZ) * z + coeff.col(XXX) * x * 3));
    // pdv_xy
    res.col(XY) += coeff.col(XXY) * x * 2;
    res.col(XY) += coeff.col(XYY) * y * 2;
    res.col(XY) += coeff.col(XYZ) * z;
    assert(res.col(XY) == coeff.col(XY) + coeff.col(XYZ) * z
        + 2 * (coeff.col(XXY) * x + coeff.col(XYY) * y));
    // pdv_xz
    res.col(XZ) += coeff.col(XXZ) * x * 2;
    res.col(XZ) += coeff.col(XZZ) * z * 2;
    res.col(XZ) += coeff.col(XYZ) * y;
    assert(res.col(XZ) == coeff.col(XZ) + coeff.col(XYZ) * y
        + 2 * (coeff.col(XXZ) * x + coeff.col(XZZ) * z));
    // pdv_yy
    res.col(YY) += coeff.col(XYY) * x;
    res.col(YY) += coeff.col(YYZ) * z;
    res.col(YY) += coeff.col(YYY) * y * 3;
    res.col(YY) += res.col(XX);
    assert(res.col(YY) == coeff.col(YYY) * y * 6
        + 2 * (coeff.col(YY) + coeff.col(XYY) * x + coeff.col(YYZ) * z));
    // pdv_yz
    res.col(YZ) += coeff.col(XYZ) * x;
    res.col(YZ) += coeff.col(YYZ) * y * 2;
    res.col(YZ) += coeff.col(YZZ) * z * 2;
    assert(res.col(YZ) == coeff.col(YZ) + coeff.col(XYZ) * x
        + 2 * (coeff.col(YYZ) * y + coeff.col(YZZ) * z));
    // pdv_zz
    res.col(ZZ) += coeff.col(XZZ) * x;
    res.col(ZZ) += coeff.col(YZZ) * y;
    res.col(ZZ) += coeff.col(ZZZ) * z * 3;
    res.col(ZZ) += res.col(ZZ);
    assert(res.col(ZZ) == coeff.col(ZZZ) * z * 6
        + 2 * (coeff.col(ZZ) + coeff.col(XZZ) * x + coeff.col(YZZ) * y));
    // pdv_xxx
    res.col(XXX) *= 6;
    assert(res.col(XXX) == coeff.col(XXX) * 6);
    // pdv_xxy
    res.col(XXY) += res.col(XXY);
    assert(res.col(XXY) == coeff.col(XXY) * 2);
    // pdv_xxz
    res.col(XXZ) += res.col(XXZ);
    assert(res.col(XXZ) == coeff.col(XXZ) * 2);
    // pdv_xyy
    res.col(XYY) += res.col(XYY);
    assert(res.col(XYY) == coeff.col(XYY) * 2);
    // pdv_xyz
    assert(res.col(XYZ) == coeff.col(XYZ));
    // pdv_xzz
    res.col(XZZ) += res.col(XZZ);
    assert(res.col(XZZ) == coeff.col(XZZ) * 2);
    // pdv_yyy
    res.col(YYY) *= 6;
    assert(res.col(YYY) == coeff.col(YYY) * 6);
    // pdv_yyz
    res.col(YYZ) += res.col(YYZ);
    assert(res.col(YYZ) == coeff.col(YYZ) * 2);
    // pdv_yzz
    res.col(YZZ) += res.col(YZZ);
    assert(res.col(YZZ) == coeff.col(YZZ) * 2);
    // pdv_zzz
    res.col(ZZZ) *= 6;
    assert(res.col(ZZZ) == coeff.col(ZZZ) * 2);
    return res;
  }

  template <int K>
  static auto GetGradValue(const Coord &xyz,
      const algebra::Matrix<Scalar, K, N> &coeff) {
    auto x = xyz[0], y = xyz[1], z = xyz[2];
    auto xx{x * x}, xy{x * y}, xz{x * z}, yy{y * y}, yz{y * z}, zz{z * z};
    algebra::Matrix<Scalar, K, 3> res;
    int i = 0;
    // pdv_x
    res.col(i) = coeff.col(X);
    res.col(i) += coeff.col(XX) * (2 * x);
    res.col(i) += coeff.col(XY) * y;
    res.col(i) += coeff.col(XZ) * z;
    res.col(i) += coeff.col(XXX) * (3 * xx);
    res.col(i) += coeff.col(XXY) * (2 * xy);
    res.col(i) += coeff.col(XXZ) * (2 * xz);
    res.col(i) += coeff.col(XYY) * yy;
    res.col(i) += coeff.col(XYZ) * yz;
    res.col(i) += coeff.col(XZZ) * zz;
    // pdv_y
    i++;
    res.col(i) = coeff.col(Y);
    res.col(i) += coeff.col(XY) * x;
    res.col(i) += coeff.col(YY) * (2 * y);
    res.col(i) += coeff.col(YZ) * z;
    res.col(i) += coeff.col(XXY) * xx;
    res.col(i) += coeff.col(XYZ) * xz;
    res.col(i) += coeff.col(XYY) * (2 * xy);
    res.col(i) += coeff.col(YYY) * (3 * yy);
    res.col(i) += coeff.col(YYZ) * (2 * yz);
    res.col(i) += coeff.col(YZZ) * zz;
    // pdv_z
    i++;
    res.col(i) = coeff.col(Z);
    res.col(i) += coeff.col(XZ) * x;
    res.col(i) += coeff.col(YZ) * y;
    res.col(i) += coeff.col(ZZ) * (2 * z);
    res.col(i) += coeff.col(XXZ) * xx;
    res.col(i) += coeff.col(XYZ) * xy;
    res.col(i) += coeff.col(XZZ) * (2 * xz);
    res.col(i) += coeff.col(YYZ) * yy;
    res.col(i) += coeff.col(YZZ) * (2 * yz);
    res.col(i) += coeff.col(ZZZ) * (3 * zz);
    return res;
  }

  template <int K>
  static auto GetSmoothness(
      const algebra::Matrix<Scalar, K, N> &integral, Scalar volume) {
    using MatKx1 = algebra::Matrix<Scalar, K, 1>;
    auto w1  // weight of 1st-order partial derivatives
        = std::pow(volume, 2./3-1);
    MatKx1 smoothness = integral.col(X);
    smoothness += integral.col(Y);
    smoothness += integral.col(Z);
    smoothness *= w1;
    auto w2  // weight of 2nd-order partial derivatives
        = std::pow(volume, 4./3-1);
    smoothness += integral.col(XX) * w2;
    smoothness += integral.col(XY) * w2;
    smoothness += integral.col(XZ) * w2;
    smoothness += integral.col(YY) * w2;
    smoothness += integral.col(YZ) * w2;
    smoothness += integral.col(ZZ) * w2;
    auto w3  // weight of 3rd-order partial derivatives
        = volume;  // = std::pow(volume, 6./3-1);
    smoothness += integral.col(XXX) * w3;
    smoothness += integral.col(XXY) * w3;
    smoothness += integral.col(XXZ) * w3;
    smoothness += integral.col(XYY) * w3;
    smoothness += integral.col(XYZ) * w3;
    smoothness += integral.col(XZZ) * w3;
    smoothness += integral.col(YYY) * w3;
    smoothness += integral.col(YYZ) * w3;
    smoothness += integral.col(YZZ) * w3;
    smoothness += integral.col(ZZZ) * w3;
    return smoothness;
  }
};

/**
 * @brief A basis of the linear space formed by polynomials less than or equal to a given degree.
 * 
 * @tparam Scalar the data type of scalar components
 * @tparam kDimensions the dimension of the underlying physical space
 * @tparam kDegrees the degree of completeness
 */
template <typename Scalar, int kDimensions, int kDegrees>
class Linear {
  using RB = Raw<Scalar, kDimensions, kDegrees>;

 public:
  static constexpr int N = RB::N;
  using Coord = typename RB::Coord;
  using MatNx1 = typename RB::MatNx1;
  using MatNxN = algebra::Matrix<Scalar, N, N>;
  using Gauss = std::conditional_t<kDimensions == 2,
      integrator::Face<Scalar, 2>, integrator::Cell<Scalar>>;

 public:
  explicit Linear(Coord const &center)
      : center_(center) {
    coeff_.setIdentity();
  }
  Linear() {
    center_.setZero();
    coeff_.setIdentity();
  }
  Linear(const Linear &) = default;
  Linear(Linear &&) noexcept = default;
  Linear &operator=(const Linear &) = default;
  Linear &operator=(Linear &&) noexcept = default;
  ~Linear() noexcept = default;

  MatNx1 operator()(Coord const &point) const {
    MatNx1 col = RB::GetValue(point - center_);
    MatNx1 res = algebra::GetLowerTriangularView(coeff_) * col;
    return res;
  }
  Coord const &center() const {
    return center_;
  }
  MatNxN const &coeff() const {
    return coeff_;
  }
  void Transform(MatNxN const &a) {
    MatNxN temp = a * coeff_;
    coeff_ = temp;
  }
  void Transform(algebra::LowerTriangularView<MatNxN> const &a) {
    MatNxN temp;
    algebra::GetLowerTriangularView(&temp) = a * coeff_;
    algebra::GetLowerTriangularView(&coeff_) = temp;
  }
  void Shift(const Coord &new_center) {
    center_ = new_center;
  }

 private:
  Coord center_;
  MatNxN coeff_;
};

template <typename Scalar, int kDimensions, int kDegrees>
class OrthoNormal {
  using RB = Raw<Scalar, kDimensions, kDegrees>;
  using LB = Linear<Scalar, kDimensions, kDegrees>;

 public:
  static constexpr int N = LB::N;
  using Coord = typename LB::Coord;
  using Gauss = typename LB::Gauss;
  using MatNx1 = typename LB::MatNx1;
  using MatNxN = typename LB::MatNxN;
  using MatNxD = algebra::Matrix<Scalar, N, kDimensions>;

 public:
  explicit OrthoNormal(const Gauss &gauss)
      : gauss_ptr_(&gauss), basis_(gauss.center()) {
    assert(gauss.PhysDim() == kDimensions);
    OrthoNormalize(&basis_, gauss);
  }
  OrthoNormal() = default;
  OrthoNormal(const OrthoNormal &) = default;
  OrthoNormal(OrthoNormal &&) noexcept = default;
  OrthoNormal &operator=(const OrthoNormal &) = default;
  OrthoNormal &operator=(OrthoNormal &&) noexcept = default;
  ~OrthoNormal() noexcept = default;

  Coord const &center() const {
    return basis_.center();
  }
  MatNxN const &coeff() const {
    return basis_.coeff();
  }
  Gauss const &GetGauss() const {
    return *gauss_ptr_;
  }
  MatNx1 operator()(const Coord &global) const {
    auto local = global;
    local -= center();
    MatNx1 col = RB::GetValue(local);
    return coeff() * col;
  }
  Scalar Measure() const {
    auto v = basis_.coeff()(0, 0);
    return 1 / (v * v);
  }
  MatNxD GetGradValue(const Coord &global) const {
    auto local = global;
    local -= center();
    return RB::GetGradValue(local, coeff());
  }

 private:
  Gauss const *gauss_ptr_;
  Linear<Scalar, kDimensions, kDegrees> basis_;
};

}  // namespace polynomial
}  // namespace mini

#endif  // MINI_POLYNOMIAL_BASIS_HPP_
