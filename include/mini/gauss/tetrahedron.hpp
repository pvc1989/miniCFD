//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_GAUSS_TETRAHEDRON_HPP_
#define MINI_GAUSS_TETRAHEDRON_HPP_

#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <utility>

#include "mini/algebra/eigen.hpp"
#include "mini/gauss/cell.hpp"

namespace mini {
namespace gauss {

/**
 * @brief 
 * 
 * @tparam Scalar 
 * @tparam kPoints 
 */
template <typename Scalar, int kPoints>
class Tetrahedron : public Cell<Scalar> {
  using Mat3x3 = algebra::Matrix<Scalar, 3, 3>;
  using Mat1x4 = algebra::Matrix<Scalar, 1, 4>;
  using Mat4x1 = algebra::Matrix<Scalar, 4, 1>;
  using Mat3x4 = algebra::Matrix<Scalar, 3, 4>;
  using Mat4x3 = algebra::Matrix<Scalar, 4, 3>;
  using Mat3x1 = algebra::Matrix<Scalar, 3, 1>;

  using Arr1x4 = algebra::Array<Scalar, 1, 4>;
  using Arr4x1 = algebra::Array<Scalar, 4, 1>;
  using Arr3x4 = algebra::Array<Scalar, 3, 4>;
  using Arr4x3 = algebra::Array<Scalar, 4, 3>;

  using Base = Cell<Scalar>;

 public:
  using typename Base::Real;
  using typename Base::LocalCoord;
  using typename Base::GlobalCoord;

 private:
  static const std::array<LocalCoord, kPoints> local_coords_;
  static const std::array<Scalar, kPoints> local_weights_;
  static const std::array<std::array<int, 3>, 4> faces_;
  Mat3x4 xyz_global_3x4_;
  std::array<GlobalCoord, kPoints> global_coords_;
  std::array<Scalar, kPoints> global_weights_;
  Scalar volume_;

 public:
  int CountVertices() const {
    return 4;
  }
  int CountQuadraturePoints() const override {
    return kPoints;
  }
  template <typename T, typename U>
  static void SortNodesOnFace(const T *cell_nodes, U *face_nodes) {
    int cnt = 0, nid = 0, sum = 0;
    while (cnt < 3) {
      auto curr_node = cell_nodes[nid];
      for (int i = 0; i < 3; ++i) {
        if (face_nodes[i] == curr_node) {
          sum += nid;
          ++cnt;
          break;
        }
      }
      ++nid;
    }
    int i_face = sum - 3;
    for (int i = 0; i < 3; ++i) {
      face_nodes[i] = cell_nodes[faces_[i_face][i]];
    }
  }

 private:
  void BuildQuadraturePoints() {
    int n = CountQuadraturePoints();
    volume_ = 0.0;
    for (int q = 0; q < n; ++q) {
      auto det_j = Jacobian(GetLocalCoord(q)).determinant();
      global_weights_[q] = local_weights_[q] * std::abs(det_j);
      volume_ += global_weights_[q];
      global_coords_[q] = LocalToGlobal(GetLocalCoord(q));
    }
  }
  static constexpr auto BuildFaces() {
    std::array<std::array<int, 3>, 4> faces{
      // Faces can be distinguished by the sum of the three minimum node ids.
      0, 2, 1/* 3 */, 0, 1, 3/* 4 */, 2, 0, 3/* 5 */, 1, 2, 3/* 6 */
    };
    return faces;
  }
  static Mat4x1 shape_4x1(Mat3x1 const &xyz_local) {
    return shape_4x1(xyz_local[0], xyz_local[1], xyz_local[2]);
  }
  static Mat4x1 shape_4x1(Scalar x_local, Scalar y_local, Scalar z_local) {
    Mat4x1 n_4x1{
      x_local, y_local, z_local, 1.0 - x_local - y_local - z_local
    };
    return n_4x1;
  }
  static Mat4x3 diff_shape_local_4x3(Scalar x_local, Scalar y_local,
      Scalar z_local) {
    Arr4x3 dn;
    dn.col(0) << 1, 0, 0, -1;
    dn.col(1) << 0, 1, 0, -1;
    dn.col(2) << 0, 0, 1, -1;
    return dn;
  }
  Mat3x3 Jacobian(Scalar x_local, Scalar y_local, Scalar z_local) const {
    return xyz_global_3x4_ * diff_shape_local_4x3(x_local, y_local, z_local);
  }
  template <typename Callable, typename MatJ>
  static Mat3x1 root(
      Callable &&func, Mat3x1 x, MatJ &&matj, Scalar xtol = 1e-5) {
    Mat3x1 res;
    do {
      res = matj(x).partialPivLu().solve(func(x));
      x -= res;
    } while (res.norm() > xtol);
    return x;
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

 public:
  explicit Tetrahedron(Mat3x4 const &xyz_global) {
    xyz_global_3x4_ = xyz_global;
    BuildQuadraturePoints();
  }
  Tetrahedron(Mat3x1 const &p0, Mat3x1 const &p1, Mat3x1 const &p2,
      Mat3x1 const &p3) {
    xyz_global_3x4_.col(0) = p0; xyz_global_3x4_.col(1) = p1;
    xyz_global_3x4_.col(2) = p2; xyz_global_3x4_.col(3) = p3;
    BuildQuadraturePoints();
  }
  Tetrahedron(std::initializer_list<Mat3x1> il) {
    assert(il.size() == 4);
    auto p = il.begin();
    for (int i = 0; i < 4; ++i) {
      xyz_global_3x4_[i] = p[i];
    }
    BuildQuadraturePoints();
  }
  Tetrahedron() {
    xyz_global_3x4_.col(0) << 0, 0, 0;
    xyz_global_3x4_.col(1) << 1, 0, 0;
    xyz_global_3x4_.col(2) << 0, 1, 0;
    xyz_global_3x4_.col(3) << 0, 0, 1;
    BuildQuadraturePoints();
  }
  Tetrahedron(const Tetrahedron &) = default;
  Tetrahedron &operator=(const Tetrahedron &) = default;
  Tetrahedron(Tetrahedron &&) noexcept = default;
  Tetrahedron &operator=(Tetrahedron &&) noexcept = default;
  virtual ~Tetrahedron() noexcept = default;

  Mat3x1 center() const override {
    Mat3x1 c = xyz_global_3x4_.col(0);
    for (int i = 1; i < 4; ++i)
      c += xyz_global_3x4_.col(i);
    c /= 4;
    return c;
  }
  Scalar volume() const override {
    return volume_;
  }
  GlobalCoord LocalToGlobal(Scalar x_local, Scalar y_local,
      Scalar z_local) const {
    return xyz_global_3x4_ * shape_4x1(x_local, y_local, z_local);
  }
  GlobalCoord LocalToGlobal(LocalCoord const &xyz_local) const override {
    return xyz_global_3x4_ * shape_4x1(xyz_local);
  }
  Mat3x3 Jacobian(const LocalCoord &xyz_local) const override {
    return Jacobian(xyz_local[0], xyz_local[1], xyz_local[2]);
  }
  LocalCoord global_to_local_3x1(Scalar x_global, Scalar y_global,
      Scalar z_global) const {
    Mat3x1 xyz_global = {x_global, y_global, z_global};
    auto func = [this, &xyz_global](Mat3x1 const &xyz_local) {
      auto res = LocalToGlobal(xyz_local);
      return res -= xyz_global;
    };
    auto jac = [this](LocalCoord const &xyz_local) {
      return Jacobian(xyz_local);
    };
    Mat3x1 xyz0 = {0, 0, 0};
    return root(func, xyz0, jac);
  }
  LocalCoord global_to_local_3x1(GlobalCoord const &xyz_global) const {
    return global_to_local_3x1(xyz_global[0], xyz_global[1], xyz_global[2]);
  }
};

template <typename Scalar, int kPoints>
class TetrahedronBuilder;

template <typename Scalar, int kPoints>
const std::array<typename Tetrahedron<Scalar, kPoints>::LocalCoord, kPoints>
Tetrahedron<Scalar, kPoints>::local_coords_
    = TetrahedronBuilder<Scalar, kPoints>::BuildLocalCoords();

template <typename Scalar, int kPoints>
const std::array<Scalar, kPoints>
Tetrahedron<Scalar, kPoints>::local_weights_
    = TetrahedronBuilder<Scalar, kPoints>::BuildLocalWeights();

template <typename Scalar, int kPoints>
const std::array<std::array<int, 3>, 4>
Tetrahedron<Scalar, kPoints>::faces_
    = Tetrahedron<Scalar, kPoints>::BuildFaces();

template <typename Scalar>
class TetrahedronBuilder<Scalar, 1> {
  static constexpr int kPoints = 1;
  using LocalCoord = typename Tetrahedron<Scalar, kPoints>::LocalCoord;

 public:
  static constexpr auto BuildLocalCoords() {
    Scalar a = 0.25;
    std::array<LocalCoord, kPoints> points;
    points[0] = { a, a, a };
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    std::array<Scalar, kPoints> weights{ 1.0 / 6.0 };
    return weights;
  }
};

template <typename Scalar>
class TetrahedronBuilder<Scalar, 4> {
  static constexpr int kPoints = 4;
  using LocalCoord = typename Tetrahedron<Scalar, kPoints>::LocalCoord;

 public:
  static constexpr auto BuildLocalCoords() {
    std::array<LocalCoord, kPoints> points;
    int q = 0;
    // the only S31 orbit
    Scalar a = 0.13819660112501051517954131656343619;
    auto c = 1 - 3 * a;
    points[q++] = { a, a, a };
    points[q++] = { a, a, c };
    points[q++] = { a, c, a };
    points[q++] = { c, a, a };
    assert(q == kPoints);
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    std::array<Scalar, kPoints> weights;
    for (int q = 0; q < kPoints; ++q)
      weights[q] = 0.25 / 6.0;
    return weights;
  }
};

template <typename Scalar>
class TetrahedronBuilder<Scalar, 14> {
  static constexpr int kPoints = 14;
  using LocalCoord = typename Tetrahedron<Scalar, kPoints>::LocalCoord;

 public:
  static constexpr auto BuildLocalCoords() {
    std::array<LocalCoord, kPoints> points;
    int q = 0;
    // the two S31 orbits
    Scalar a_s31[] = {
        0.31088591926330060979734573376345783,
        0.09273525031089122640232391373703061 };
    for (auto a : a_s31) {
      auto c = 1 - 3 * a;
      points[q++] = { a, a, a };
      points[q++] = { a, a, c };
      points[q++] = { a, c, a };
      points[q++] = { c, a, a };
    }
    {  // the only S22 orbit
      Scalar a = 0.04550370412564964949188052627933943;
      auto c = (1 - 2 * a) * .5;
      points[q++] = { a, a, c };
      points[q++] = { a, c, a };
      points[q++] = { a, c, c };
      points[q++] = { c, a, a };
      points[q++] = { c, a, c };
      points[q++] = { c, c, a };
    }
    assert(q == kPoints);
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    std::array<Scalar, kPoints> weights;
    for (int q = 0; q < 4; ++q)
      weights[q] = 0.11268792571801585079918565233328633;
    for (int q = 4; q < 8; ++q)
      weights[q] = 0.07349304311636194954371020548632750;
    for (int q = 8; q < kPoints; ++q)
      weights[q] = 0.04254602077708146643806942812025744;
    for (int q = 0; q < kPoints; ++q)
      weights[q] /= 6.0;
    return weights;
  }
};

template <typename Scalar>
class TetrahedronBuilder<Scalar, 15> {
  static constexpr int kPoints = 15;
  using LocalCoord = typename Tetrahedron<Scalar, kPoints>::LocalCoord;

 public:
  static constexpr auto BuildLocalCoords() {
    std::array<LocalCoord, kPoints> points;
    int q = 0;
    {  // the only S4 orbit
      Scalar a = 0.25;
      points[q++] = { a, a, a };
    }
    // the two S31 orbits
    Scalar a_s31[] = {
        0.09197107805272303,
        0.31979362782962991 };
    for (auto a : a_s31) {
      auto c = 1 - 3 * a;
      points[q++] = { a, a, a };
      points[q++] = { a, a, c };
      points[q++] = { a, c, a };
      points[q++] = { c, a, a };
    }
    {  // the only S22 orbit
      Scalar a = 0.05635083268962916;
      auto c = (1 - 2 * a) * .5;
      points[q++] = { a, a, c };
      points[q++] = { a, c, a };
      points[q++] = { a, c, c };
      points[q++] = { c, a, a };
      points[q++] = { c, a, c };
      points[q++] = { c, c, a };
    }
    assert(q == kPoints);
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    std::array<Scalar, kPoints> weights;
    for (int q = 0; q < 1; ++q)
      weights[q] = 16.0 / 135.0;
    for (int q = 1; q < 5; ++q)
      weights[q] = 0.07193708377901862;
    for (int q = 5; q < 9; ++q)
      weights[q] = 0.06906820722627239;
    for (int q = 9; q < kPoints; ++q)
      weights[q] = 20.0 / 378.0;
    for (int q = 0; q < kPoints; ++q)
      weights[q] /= 6.0;
    return weights;
  }
};

template <typename Scalar>
class TetrahedronBuilder<Scalar, 24> {
  static constexpr int kPoints = 24;
  using LocalCoord = typename Tetrahedron<Scalar, kPoints>::LocalCoord;

 public:
  static constexpr auto BuildLocalCoords() {
    std::array<LocalCoord, kPoints> points;
    int q = 0;
    // the three S31 orbits
    Scalar a_s31[] = {
        0.21460287125915202928883921938628499,
        0.04067395853461135311557944895641006,
        0.32233789014227551034399447076249213 };
    for (auto a : a_s31) {
      auto c = 1 - 3 * a;
      points[q++] = { a, a, a };
      points[q++] = { a, a, c };
      points[q++] = { a, c, a };
      points[q++] = { c, a, a };
    }
    {  // the only S211 orbit
      Scalar a = 0.06366100187501752529923552760572698;
      Scalar b = 0.60300566479164914136743113906093969;
      auto c = 1 - a - a - b;
      points[q++] = { a, a, b };
      points[q++] = { a, a, c };
      points[q++] = { a, b, a };
      points[q++] = { a, b, c };
      points[q++] = { a, c, a };
      points[q++] = { a, c, b };
      points[q++] = { b, a, a };
      points[q++] = { b, a, c };
      points[q++] = { b, c, a };
      points[q++] = { c, a, a };
      points[q++] = { c, a, b };
      points[q++] = { c, b, a };
    }
    assert(q == kPoints);
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    std::array<Scalar, kPoints> weights;
    for (int q = 0; q < 4; ++q)
      weights[q] = 0.03992275025816749209969062755747998;
    for (int q = 4; q < 8; ++q)
      weights[q] = 0.01007721105532064294801323744593686;
    for (int q = 8; q < 12; ++q)
      weights[q] = 0.05535718154365472209515327785372602;
    for (int q = 12; q < 24; ++q)
      weights[q] = 27./560.;
    for (int q = 0; q < 24; ++q)
      weights[q] /= 6.0;
    return weights;
  }
};

template <typename Scalar>
class TetrahedronBuilder<Scalar, 46> {
  static constexpr int kPoints = 46;
  using LocalCoord = typename Tetrahedron<Scalar, kPoints>::LocalCoord;

 public:
  static constexpr auto BuildLocalCoords() {
    std::array<LocalCoord, kPoints> points;
    int q = 0;
    // the four S31 orbits
    Scalar a_s31[] = {
        .03967542307038990126507132953938949,
        .31448780069809631378416056269714830,
        .10198669306270330000000000000000000,
        .18420369694919151227594641734890918 };
    for (auto a : a_s31) {
      auto c = 1 - 3 * a;
      points[q++] = { a, a, a };
      points[q++] = { a, a, c };
      points[q++] = { a, c, a };
      points[q++] = { c, a, a };
    }
    {  // the only S22 orbit
      Scalar a = .06343628775453989240514123870189827;
      auto c = (1 - 2 * a) * .5;
      points[q++] = { a, a, c };
      points[q++] = { a, c, a };
      points[q++] = { a, c, c };
      points[q++] = { c, a, a };
      points[q++] = { c, a, c };
      points[q++] = { c, c, a };
    }
    // the two S211 orbits
    std::pair<Scalar, Scalar> ab_s211[2]{
      { .02169016206772800480266248262493018,
        .71993192203946593588943495335273478 },
      { .20448008063679571424133557487274534,
        .58057719012880922417539817139062041 }};
    for (auto [a, b] : ab_s211) {
      auto c = 1 - a - a - b;
      points[q++] = { a, a, b };
      points[q++] = { a, a, c };
      points[q++] = { a, b, a };
      points[q++] = { a, b, c };
      points[q++] = { a, c, a };
      points[q++] = { a, c, b };
      points[q++] = { b, a, a };
      points[q++] = { b, a, c };
      points[q++] = { b, c, a };
      points[q++] = { c, a, a };
      points[q++] = { c, a, b };
      points[q++] = { c, b, a };
    }
    assert(q == kPoints);
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    std::array<Scalar, kPoints> weights;
    for (int q = 0; q < 4; ++q)
      weights[q] = .00639714777990232132145142033517302;
    for (int q = 4; q < 8; ++q)
      weights[q] = .04019044802096617248816115847981783;
    for (int q = 8; q < 12; ++q)
      weights[q] = .02430797550477032117486910877192260;
    for (int q = 12; q < 16; ++q)
      weights[q] = .05485889241369744046692412399039144;
    for (int q = 16; q < 22; ++q)
      weights[q] = .03571961223409918246495096899661762;
    for (int q = 22; q < 34; ++q)
      weights[q] = .00718319069785253940945110521980376;
    for (int q = 34; q < 46; ++q)
      weights[q] = .01637218194531911754093813975611913;
    for (int q = 0; q < 46; ++q)
      weights[q] /= 6.0;
    return weights;
  }
};

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_TETRAHEDRON_HPP_
