//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_GAUSS_TRIANGLE_HPP_
#define MINI_GAUSS_TRIANGLE_HPP_

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstring>
#include <type_traits>

#include "mini/algebra/eigen.hpp"
#include "mini/gauss/face.hpp"

namespace mini {
namespace gauss {

/**
 * @brief 
 * 
 * @tparam Scalar 
 * @tparam kPoints 
 */
template <std::floating_point Scalar, int kDimensions, int kPoints>
class Triangle : public Face<Scalar, kDimensions> {
  static constexpr int D = kDimensions;
  using Arr1x3 = algebra::Array<Scalar, 1, 3>;
  using Arr3x1 = algebra::Array<Scalar, 3, 1>;
  using Arr3x2 = algebra::Array<Scalar, 3, 2>;

  using MatDx3 = algebra::Matrix<Scalar, D, 3>;
  using MatDx2 = algebra::Matrix<Scalar, D, 2>;
  using MatDx1 = algebra::Matrix<Scalar, D, 1>;
  using MatDxD = algebra::Matrix<Scalar, D, D>;
  using Mat3x1 = algebra::Matrix<Scalar, 3, 1>;
  using Mat3x2 = algebra::Matrix<Scalar, 3, 2>;
  using Mat2x1 = algebra::Matrix<Scalar, 2, 1>;

 public:
  using Real = Scalar;
  using LocalCoord = Mat2x1;
  using GlobalCoord = MatDx1;

 private:
  static const std::array<Scalar, kPoints> local_weights_;
  static const std::array<LocalCoord, kPoints> local_coords_;
  std::array<GlobalCoord, 3> xyz_global_Dx3_;
  std::array<Scalar, kPoints> global_weights_;
  std::array<GlobalCoord, kPoints> global_coords_;
  std::array<MatDxD, kPoints> normal_frames_;
  Scalar area_;

 public:
  int CountVertices() const override {
    return 3;
  }
  const GlobalCoord &GetVertex(int i) const override {
    return xyz_global_Dx3_[i];
  }
  int CountQuadraturePoints() const override {
    return kPoints;
  }
  void BuildNormalFrames() {
    static_assert(D == 3);
    int n = CountQuadraturePoints();
    for (int q = 0; q < n; ++q) {
      MatDx2 dr = Jacobian(GetLocalCoord(q));
      auto &frame = normal_frames_[q];
      frame.col(0) = dr.col(0).cross(dr.col(1)).normalized();
      frame.col(2) = dr.col(1).normalized();
      frame.col(1) = frame.col(2).cross(frame.col(0));
    }
  }

 private:
  void BuildQuadraturePoints() {
    int n = CountQuadraturePoints();
    area_ = 0.0;
    for (int q = 0; q < n; ++q) {
      auto mat_j = Jacobian(GetLocalCoord(q));
      auto det_j = this->CellDim() < this->PhysDim()
          ? std::sqrt((mat_j.transpose() * mat_j).determinant())
          : mat_j.determinant();
      global_weights_[q] = local_weights_[q] * std::abs(det_j);
      area_ += global_weights_[q];
      global_coords_[q] = LocalToGlobal(GetLocalCoord(q));
    }
  }
  static Mat3x1 shape_3x1(Mat2x1 const &xy_local) {
    return shape_3x1(xy_local[0], xy_local[1]);
  }
  static Mat3x1 shape_3x1(Scalar x_local, Scalar y_local) {
    Mat3x1 n_3x1{
      x_local, y_local, 1.0 - x_local - y_local
    };
    return n_3x1;
  }
  static Mat3x2 diff_shape_local_3x2(Scalar x_local, Scalar y_local) {
    Arr3x2 dn;
    dn.col(0) << 1, 0, -1;
    dn.col(1) << 0, 1, -1;
    return dn;
  }
  MatDx2 Jacobian(Scalar x_local, Scalar y_local) const {
    auto dn = diff_shape_local_3x2(x_local, y_local);
    MatDx2 dr = xyz_global_Dx3_[0] * dn.row(0);
    dr +=  xyz_global_Dx3_[1] * dn.row(1);
    dr +=  xyz_global_Dx3_[2] * dn.row(2);
    return dr;
  }

 public:
  GlobalCoord const &GetGlobalCoord(int q) const override {
    return global_coords_[q];
  }
  Scalar const &GetGlobalWeight(int q) const override {
    return global_weights_[q];
  }
  LocalCoord const &GetLocalCoord(int q) const override {
    return local_coords_[q];
  }
  Scalar const &GetLocalWeight(int q) const override {
    return local_weights_[q];
  }
  GlobalCoord LocalToGlobal(Scalar x_local, Scalar y_local) const {
    auto shape = shape_3x1(x_local, y_local);
    GlobalCoord product = xyz_global_Dx3_[0] * shape[0];
    product += xyz_global_Dx3_[1] * shape[1];
    product += xyz_global_Dx3_[2] * shape[2];
    return product;
  }
  GlobalCoord LocalToGlobal(LocalCoord const &xy_local) const override {
    return LocalToGlobal(xy_local[0], xy_local[1]);
  }
  MatDx2 Jacobian(const LocalCoord &xy_local) const override {
    return Jacobian(xy_local[0], xy_local[1]);
  }
  GlobalCoord center() const override {
    GlobalCoord c = xyz_global_Dx3_[0];
    c += xyz_global_Dx3_[1];
    c += xyz_global_Dx3_[2];
    c /= 3;
    return c;
  }
  Scalar area() const override {
    return area_;
  }
  const MatDxD &GetNormalFrame(int q) const override {
    return normal_frames_[q];
  }

 public:
  explicit Triangle(MatDx3 const &xyz_global) {
    xyz_global_Dx3_[0] = xyz_global.col(0);
    xyz_global_Dx3_[1] = xyz_global.col(1);
    xyz_global_Dx3_[2] = xyz_global.col(2);
    BuildQuadraturePoints();
  }
  Triangle(MatDx1 const &p0, MatDx1 const &p1, MatDx1 const &p2) {
    xyz_global_Dx3_[0] = p0;
    xyz_global_Dx3_[1] = p1;
    xyz_global_Dx3_[2] = p2;
    BuildQuadraturePoints();
  }
  Triangle(std::initializer_list<MatDx1> il) {
    assert(il.size() == 3);
    auto p = il.begin();
    xyz_global_Dx3_[0] = p[0];
    xyz_global_Dx3_[1] = p[1];
    xyz_global_Dx3_[2] = p[2];
    BuildQuadraturePoints();
  }
  Triangle(const Triangle &) = default;
  Triangle &operator=(const Triangle &) = default;
  Triangle(Triangle &&) noexcept = default;
  Triangle &operator=(Triangle &&) noexcept = default;
  virtual ~Triangle() noexcept = default;
};

template <std::floating_point Scalar, int kDimensions, int kPoints>
class TriangleBuilder;

template <std::floating_point Scalar, int kDimensions, int kPoints>
std::array<typename Triangle<Scalar, kDimensions, kPoints>::LocalCoord,
    kPoints> const
Triangle<Scalar, kDimensions, kPoints>::local_coords_
    = TriangleBuilder<Scalar, kDimensions, kPoints>::BuildLocalCoords();

template <std::floating_point Scalar, int kDimensions, int kPoints>
const std::array<Scalar, kPoints>
Triangle<Scalar, kDimensions, kPoints>::local_weights_
    = TriangleBuilder<Scalar, kDimensions, kPoints>::BuildLocalWeights();

template <std::floating_point Scalar, int kDimensions>
class TriangleBuilder<Scalar, kDimensions, 1> {
  static constexpr int kPoints = 1;
  using LocalCoord =
      typename Triangle<Scalar, kDimensions, kPoints>::LocalCoord;

 public:
  static constexpr auto BuildLocalCoords() {
    Scalar a = .3333333333333333333333333333333333;
    std::array<LocalCoord, kPoints> points;
    points[0] = { a, a };
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    std::array<Scalar, kPoints> weights{ 1.0 / 2.0 };
    return weights;
  }
};

template <std::floating_point Scalar, int kDimensions>
class TriangleBuilder<Scalar, kDimensions, 3> {
  static constexpr int kPoints = 3;
  using LocalCoord =
      typename Triangle<Scalar, kDimensions, kPoints>::LocalCoord;

 public:
  static constexpr auto BuildLocalCoords() {
    std::array<LocalCoord, kPoints> points;
    int q = 0;
    // the only S21 orbits
    Scalar a_s21[] = { 1./6. };
    for (auto a : a_s21) {
      auto b = 1 - a - a;
      points[q++] = { a, a };
      points[q++] = { a, b };
      points[q++] = { b, a };
    }
    assert(q == kPoints);
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    std::array<Scalar, kPoints> weights;
    for (int q = 0; q < 3; ++q)
      weights[q] = 1./3.;
    for (int q = 0; q < kPoints; ++q)
      weights[q] /= 2.0;
    return weights;
  }
};

template <std::floating_point Scalar, int kDimensions>
class TriangleBuilder<Scalar, kDimensions, 6> {
  static constexpr int kPoints = 6;
  using LocalCoord =
      typename Triangle<Scalar, kDimensions, kPoints>::LocalCoord;

 public:
  static constexpr auto BuildLocalCoords() {
    std::array<LocalCoord, kPoints> points;
    int q = 0;
    // the two S21 orbits
    Scalar a_s21[] = {
        .44594849091596488631832925388305199,
        .09157621350977074345957146340220151 };
    for (auto a : a_s21) {
      auto b = 1 - a - a;
      points[q++] = { a, a };
      points[q++] = { a, b };
      points[q++] = { b, a };
    }
    assert(q == kPoints);
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    std::array<Scalar, kPoints> weights;
    for (int q = 0; q < 3; ++q)
      weights[q] = .22338158967801146569500700843312280;
    for (int q = 3; q < 6; ++q)
      weights[q] = .10995174365532186763832632490021053;
    for (int q = 0; q < kPoints; ++q)
      weights[q] /= 2.0;
    return weights;
  }
};

template <std::floating_point Scalar, int kDimensions>
class TriangleBuilder<Scalar, kDimensions, 12> {
  static constexpr int kPoints = 12;
  using LocalCoord =
      typename Triangle<Scalar, kDimensions, kPoints>::LocalCoord;

 public:
  static constexpr auto BuildLocalCoords() {
    std::array<LocalCoord, kPoints> points;
    int q = 0;
    // the two S21 orbits
    Scalar a_s21[] = {
        .06308901449150222834033160287081916,
        .24928674517091042129163855310701908 };
    for (auto a : a_s21) {
      auto b = 1 - a - a;
      points[q++] = { a, a };
      points[q++] = { a, b };
      points[q++] = { b, a };
    }
    {  // the only S111 orbit
      Scalar a = .05314504984481694735324967163139815;
      Scalar b = .31035245103378440541660773395655215;
      Scalar c = 1 - a - b;
      points[q++] = { a, b };
      points[q++] = { a, c };
      points[q++] = { b, a };
      points[q++] = { b, c };
      points[q++] = { c, a };
      points[q++] = { c, b };
    }
    assert(q == kPoints);
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    std::array<Scalar, kPoints> weights;
    for (int q = 0; q < 3; ++q)
      weights[q] = .05084490637020681692093680910686898;
    for (int q = 3; q < 6; ++q)
      weights[q] = .11678627572637936602528961138557944;
    for (int q = 6; q < 12; ++q)
      weights[q] = .08285107561837357519355345642044245;
    for (int q = 0; q < kPoints; ++q)
      weights[q] /= 2.0;
    return weights;
  }
};

template <std::floating_point Scalar, int kDimensions>
class TriangleBuilder<Scalar, kDimensions, 16> {
  static constexpr int kPoints = 16;
  using LocalCoord =
      typename Triangle<Scalar, kDimensions, kPoints>::LocalCoord;

 public:
  static constexpr auto BuildLocalCoords() {
    std::array<LocalCoord, kPoints> points;
    int q = 0;
    {  // the only S3 orbit
      Scalar a = .33333333333333333333333333333333333;
      points[q++] = { a, a };
    }
    // the three S21 orbits
    Scalar a_s21[] = {
        .17056930775176020662229350149146450,
        .05054722831703097545842355059659895,
        .45929258829272315602881551449416932 };
    for (auto a : a_s21) {
      auto b = 1 - a - a;
      points[q++] = { a, a };
      points[q++] = { a, b };
      points[q++] = { b, a };
    }
    {  // the six S111 orbits
      Scalar a = .26311282963463811342178578628464359;
      Scalar b = .00839477740995760533721383453929445;
      Scalar c = 1 - a - b;
      points[q++] = { a, b };
      points[q++] = { a, c };
      points[q++] = { b, a };
      points[q++] = { b, c };
      points[q++] = { c, a };
      points[q++] = { c, b };
    }
    assert(q == kPoints);
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    std::array<Scalar, kPoints> weights;
    for (int q = 0; q < 1; ++q)
      weights[q] = .14431560767778716825109111048906462;
    for (int q = 1; q < 4; ++q)
      weights[q] = .10321737053471825028179155029212903;
    for (int q = 4; q < 7; ++q)
      weights[q] = .03245849762319808031092592834178060;
    for (int q = 7; q < 10; ++q)
      weights[q] = .09509163426728462479389610438858432;
    for (int q = 10; q < 16; ++q)
      weights[q] = .02723031417443499426484469007390892;
    for (int q = 0; q < 16; ++q)
      weights[q] /= 2.0;
    return weights;
  }
};

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_TRIANGLE_HPP_
