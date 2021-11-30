//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_INTEGRATOR_TRI_HPP_
#define MINI_INTEGRATOR_TRI_HPP_

#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>

#include "mini/algebra/eigen.hpp"
#include "mini/integrator/face.hpp"

namespace mini {
namespace integrator {

/**
 * @brief 
 * 
 * @tparam Scalar 
 * @tparam kQuad 
 */
template <typename Scalar, int kDim, int kQuad>
class Tri : public Face<Scalar, kDim> {
  static constexpr int D = kDim;
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
  static const std::array<Scalar, kQuad> local_weights_;
  static const std::array<LocalCoord, kQuad> local_coords_;
  MatDx3 xyz_global_Dx3_;
  std::array<Scalar, kQuad> global_weights_;
  std::array<GlobalCoord, kQuad> global_coords_;
  std::array<MatDxD, kQuad> normal_frames_;
  Scalar area_;

 public:
  int CountQuadPoints() const override {
    return kQuad;
  }
  void BuildNormalFrames() {
    static_assert(D == 3);
    int n = CountQuadPoints();
    for (int i = 0; i < n; ++i) {
      auto& local = GetLocalCoord(i);
      auto dn = diff_shape_local_3x2(local[0], local[1]);
      MatDx2 dr = xyz_global_Dx3_ * dn;
      auto& frame = normal_frames_[i];
      frame.col(0) = dr.col(0).cross(dr.col(1)).normalized();
      frame.col(2) = dr.col(1).normalized();
      frame.col(1) = frame.col(2).cross(frame.col(0));
    }
  }

 private:
  void BuildQuadPoints() {
    int n = CountQuadPoints();
    area_ = 0.0;
    for (int i = 0; i < n; ++i) {
      auto mat_j = Jacobian(GetLocalCoord(i));
      auto det_j = this->CellDim() < this->PhysDim()
          ? std::sqrt((mat_j.transpose() * mat_j).determinant())
          : mat_j.determinant();
      global_weights_[i] = local_weights_[i] * std::abs(det_j);
      area_ += global_weights_[i];
      global_coords_[i] = LocalToGlobal(GetLocalCoord(i));
    }
  }
  static Mat3x1 shape_3x1(Mat2x1 const& xy_local) {
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
    return xyz_global_Dx3_ * diff_shape_local_3x2(x_local, y_local);
  }

 public:
  GlobalCoord const& GetGlobalCoord(int i) const override {
    return global_coords_[i];
  }
  Scalar const& GetGlobalWeight(int i) const override {
    return global_weights_[i];
  }
  LocalCoord const& GetLocalCoord(int i) const override {
    return local_coords_[i];
  }
  Scalar const& GetLocalWeight(int i) const override {
    return local_weights_[i];
  }
  GlobalCoord LocalToGlobal(Scalar x_local, Scalar y_local) const {
    return xyz_global_Dx3_ * shape_3x1(x_local, y_local);
  }
  GlobalCoord LocalToGlobal(LocalCoord const& xy_local) const override {
    return xyz_global_Dx3_ * shape_3x1(xy_local);
  }
  MatDx2 Jacobian(const LocalCoord& xy_local) const override {
    return Jacobian(xy_local[0], xy_local[1]);
  }
  MatDx1 center() const override {
    MatDx1 c = xyz_global_Dx3_.col(0);
    for (int i = 1; i < 3; ++i)
      c += xyz_global_Dx3_.col(i);
    c /= 3;
    return c;
  }
  Scalar area() const override {
    return area_;
  }
  const MatDxD& GetNormalFrame(int i) const override {
    return normal_frames_[i];
  }

 public:
  explicit Tri(MatDx3 const& xyz_global) {
    xyz_global_Dx3_ = xyz_global;
    BuildQuadPoints();
  }
  Tri(MatDx1 const& p0, MatDx1 const& p1, MatDx1 const& p2) {
    xyz_global_Dx3_.col(0) = p0;
    xyz_global_Dx3_.col(1) = p1;
    xyz_global_Dx3_.col(2) = p2;
    BuildQuadPoints();
  }
  Tri(std::initializer_list<MatDx1> il) {
    assert(il.size() == 3);
    auto p = il.begin();
    for (int i = 0; i < 3; ++i) {
      xyz_global_Dx3_[i] = p[i];
    }
    BuildQuadPoints();
  }
  Tri(const Tri&) = default;
  Tri& operator=(const Tri&) = default;
  Tri(Tri&&) noexcept = default;
  Tri& operator=(Tri&&) noexcept = default;
  virtual ~Tri() noexcept = default;
};

template <typename Scalar, int kDim, int kQuad>
class TriBuilder;

template <typename Scalar, int kDim, int kQuad>
const std::array<typename Tri<Scalar, kDim, kQuad>::LocalCoord, kQuad>
Tri<Scalar, kDim, kQuad>::local_coords_
    = TriBuilder<Scalar, kDim, kQuad>::BuildLocalCoords();

template <typename Scalar, int kDim, int kQuad>
const std::array<Scalar, kQuad>
Tri<Scalar, kDim, kQuad>::local_weights_
    = TriBuilder<Scalar, kDim, kQuad>::BuildLocalWeights();

template <typename Scalar, int kDim>
class TriBuilder<Scalar, kDim, 16> {
  static constexpr int kQuad = 16;
  using LocalCoord = typename Tri<Scalar, kDim, kQuad>::LocalCoord;

 public:
  static constexpr auto BuildLocalCoords() {
    std::array<LocalCoord, kQuad> points;
    int i = 0;
    {  // the only S3 orbit
      Scalar a = .3333333333333333333333333333333333;
      points[i++] = { a, a };
    }
    // the three S21 orbits
    Scalar a_s21[] = {
        .1705693077517602066222935014914645,
        .0505472283170309754584235505965989,
        .4592925882927231560288155144941693 };
    for (int i = 0; i < 3; ++i) {
      auto a = a_s21[i];
      auto b = 1 - a - a;
      points[i++] = { a, a };
      points[i++] = { a, b };
      points[i++] = { b, a };
    }
    {  // the six S111 orbits
      Scalar a = .2631128296346381134217857862846436;
      Scalar b = .0083947774099576053372138345392944;
      Scalar c = 1 - a - b;
      points[i++] = { a, b };
      points[i++] = { a, c };
      points[i++] = { b, a };
      points[i++] = { b, c };
      points[i++] = { c, a };
      points[i++] = { c, b };
    }
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    std::array<Scalar, kQuad> weights;
    for (int i = 0; i < 1; ++i)
      weights[i] = .1443156076777871682510911104890646;
    for (int i = 1; i < 4; ++i)
      weights[i] = .1032173705347182502817915502921290;
    for (int i = 4; i < 7; ++i)
      weights[i] = .0324584976231980803109259283417806;
    for (int i = 7; i < 10; ++i)
      weights[i] = .0950916342672846247938961043885843;
    for (int i = 10; i < 16; ++i)
      weights[i] = .0272303141744349942648446900739089;
    for (int i = 0; i < 16; ++i)
      weights[i] /= 2.0;
    return weights;
  }
};

}  // namespace integrator
}  // namespace mini

#endif  // MINI_INTEGRATOR_TRI_HPP_
