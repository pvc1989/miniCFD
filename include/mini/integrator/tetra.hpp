//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_INTEGRATOR_TETRA_HPP_
#define MINI_INTEGRATOR_TETRA_HPP_

#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>

#include "mini/algebra/eigen.hpp"
#include "mini/integrator/cell.hpp"

namespace mini {
namespace integrator {

/**
 * @brief 
 * 
 * @tparam Scalar 
 * @tparam kQuad 
 */
template <typename Scalar, int kQuad>
class Tetra : public Cell<Scalar> {
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
  static const std::array<LocalCoord, kQuad> local_coords_;
  static const std::array<Scalar, kQuad> local_weights_;
  static const std::array<std::array<int, 3>, 4> faces_;
  Mat3x4 xyz_global_3x4_;
  std::array<GlobalCoord, kQuad> global_coords_;
  std::array<Scalar, kQuad> global_weights_;
  Scalar volume_;

 public:
  int CountQuadPoints() const override {
    return kQuad;
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
  void BuildQuadPoints() {
    int n = CountQuadPoints();
    volume_ = 0.0;
    for (int i = 0; i < n; ++i) {
      auto det_j = Jacobian(GetLocalCoord(i)).determinant();
      global_weights_[i] = local_weights_[i] * std::abs(det_j);
      volume_ += global_weights_[i];
      global_coords_[i] = LocalToGlobal(GetLocalCoord(i));
    }
  }
  static constexpr auto BuildFaces() {
    std::array<std::array<int, 3>, 4> faces{
      // Faces can be distinguished by the sum of the three minimum node ids.
      0, 2, 1/* 3 */, 0, 1, 3/* 4 */, 2, 0, 3/* 5 */, 1, 2, 3/* 6 */
    };
    return faces;
  }
  static Mat4x1 shape_4x1(Mat3x1 const& xyz_local) {
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
      Callable&& func, Mat3x1 x, MatJ&& matj, Scalar xtol = 1e-5) {
    Mat3x1 res;
    do {
      res = matj(x).partialPivLu().solve(func(x));
      x -= res;
    } while (res.norm() > xtol);
    return x;
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

 public:
  explicit Tetra(Mat3x4 const& xyz_global) {
    xyz_global_3x4_ = xyz_global;
    BuildQuadPoints();
  }
  Tetra(Mat3x1 const& p0, Mat3x1 const& p1, Mat3x1 const& p2,
      Mat3x1 const& p3) {
    xyz_global_3x4_.col(0) = p0; xyz_global_3x4_.col(1) = p1;
    xyz_global_3x4_.col(2) = p2; xyz_global_3x4_.col(3) = p3;
    BuildQuadPoints();
  }
  Tetra(std::initializer_list<Mat3x1> il) {
    assert(il.size() == 4);
    auto p = il.begin();
    for (int i = 0; i < 4; ++i) {
      xyz_global_3x4_[i] = p[i];
    }
    BuildQuadPoints();
  }
  Tetra() {
    xyz_global_3x4_.col(0) << 0, 0, 0;
    xyz_global_3x4_.col(1) << 1, 0, 0;
    xyz_global_3x4_.col(2) << 0, 1, 0;
    xyz_global_3x4_.col(3) << 0, 0, 1;
    BuildQuadPoints();
  }
  Tetra(const Tetra&) = default;
  Tetra& operator=(const Tetra&) = default;
  Tetra(Tetra&&) noexcept = default;
  Tetra& operator=(Tetra&&) noexcept = default;
  virtual ~Tetra() noexcept = default;

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
  GlobalCoord LocalToGlobal(LocalCoord const& xyz_local) const override {
    return xyz_global_3x4_ * shape_4x1(xyz_local);
  }
  Mat3x3 Jacobian(const LocalCoord& xyz_local) const override {
    return Jacobian(xyz_local[0], xyz_local[1], xyz_local[2]);
  }
  LocalCoord global_to_local_3x1(Scalar x_global, Scalar y_global,
      Scalar z_global) const {
    Mat3x1 xyz_global = {x_global, y_global, z_global};
    auto func = [this, &xyz_global](Mat3x1 const& xyz_local) {
      auto res = LocalToGlobal(xyz_local);
      return res -= xyz_global;
    };
    auto jac = [this](LocalCoord const& xyz_local) {
      return Jacobian(xyz_local);
    };
    Mat3x1 xyz0 = {0, 0, 0};
    return root(func, xyz0, jac);
  }
  LocalCoord global_to_local_3x1(GlobalCoord const& xyz_global) const {
    return global_to_local_3x1(xyz_global[0], xyz_global[1], xyz_global[2]);
  }
};

template <typename Scalar, int kQuad>
class TetraBuilder;

template <typename Scalar, int kQuad>
const std::array<typename Tetra<Scalar, kQuad>::LocalCoord, kQuad>
Tetra<Scalar, kQuad>::local_coords_
    = TetraBuilder<Scalar, kQuad>::BuildLocalCoords();

template <typename Scalar, int kQuad>
const std::array<Scalar, kQuad>
Tetra<Scalar, kQuad>::local_weights_
    = TetraBuilder<Scalar, kQuad>::BuildLocalWeights();

template <typename Scalar, int kQuad>
const std::array<std::array<int, 3>, 4>
Tetra<Scalar, kQuad>::faces_
    = Tetra<Scalar, kQuad>::BuildFaces();

template <typename Scalar>
class TetraBuilder<Scalar, 24> {
  static constexpr int kQuad = 24;
  using LocalCoord = typename Tetra<Scalar, kQuad>::LocalCoord;

 public:
  static constexpr auto BuildLocalCoords() {
    std::array<LocalCoord, kQuad> points;
    int i = 0;
    // the three S31 orbits
    Scalar a_s31[] = {
        0.214602871259152,
        0.04067395853461135,
        0.3223378901422755 };
    for (auto a : a_s31) {
      auto c = 1 - 3 * a;
      points[i++] = { a, a, a };
      points[i++] = { a, a, c };
      points[i++] = { a, c, a };
      points[i++] = { c, a, a };
    }
    {  // the only S211 orbit
      Scalar a = 0.06366100187501753, b = 0.6030056647916492;
      auto c = 1 - a - a - b;
      points[i++] = { a, a, b };
      points[i++] = { a, a, c };
      points[i++] = { a, b, a };
      points[i++] = { a, b, c };
      points[i++] = { a, c, a };
      points[i++] = { a, c, b };
      points[i++] = { b, a, a };
      points[i++] = { b, a, c };
      points[i++] = { b, c, a };
      points[i++] = { c, a, a };
      points[i++] = { c, a, b };
      points[i++] = { c, b, a };
    }
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    std::array<Scalar, kQuad> weights;
    for (int i = 0; i < 4; ++i)
      weights[i] = 0.03992275025816749;
    for (int i = 4; i < 8; ++i)
      weights[i] = 0.01007721105532064;
    for (int i = 8; i < 12; ++i)
      weights[i] = 0.05535718154365473;
    for (int i = 12; i < 24; ++i)
      weights[i] = 0.04821428571428571;
    for (int i = 0; i < 24; ++i)
      weights[i] /= 6.0;
    return weights;
  }
};

template <typename Scalar>
class TetraBuilder<Scalar, 46> {
  static constexpr int kQuad = 46;
  using LocalCoord = typename Tetra<Scalar, kQuad>::LocalCoord;

 public:
  static constexpr auto BuildLocalCoords() {
    std::array<LocalCoord, kQuad> points;
    int i = 0;
    // the four S31 orbits
    Scalar a_s31[] = {
        .0396754230703899012650713295393895,
        .3144878006980963137841605626971483,
        .1019866930627033000000000000000000,
        .1842036969491915122759464173489092 };
    for (auto a : a_s31) {
      auto c = 1 - 3 * a;
      points[i++] = { a, a, a };
      points[i++] = { a, a, c };
      points[i++] = { a, c, a };
      points[i++] = { c, a, a };
    }
    {  // the only S22 orbit
      Scalar a = .0634362877545398924051412387018983;
      auto c = 1 - 2 * a;
      points[i++] = { a, a, c };
      points[i++] = { a, c, a };
      points[i++] = { a, c, c };
      points[i++] = { c, a, a };
      points[i++] = { c, a, c };
      points[i++] = { c, c, a };
    }
    // the two S211 orbits
    Scalar ab_s211[][2] = {
      { .0216901620677280048026624826249302,
        .7199319220394659358894349533527348 },
      { .2044800806367957142413355748727453,
        .5805771901288092241753981713906204 }};
    for (int i = 0; i < 2; ++i) {
      auto a = ab_s211[i][0];
      auto b = ab_s211[i][1];
      auto c = 1 - a - a - b;
      points[i++] = { a, a, b };
      points[i++] = { a, a, c };
      points[i++] = { a, b, a };
      points[i++] = { a, b, c };
      points[i++] = { a, c, a };
      points[i++] = { a, c, b };
      points[i++] = { b, a, a };
      points[i++] = { b, a, c };
      points[i++] = { b, c, a };
      points[i++] = { c, a, a };
      points[i++] = { c, a, b };
      points[i++] = { c, b, a };
    }
    return points;
  }
  static constexpr auto BuildLocalWeights() {
    std::array<Scalar, kQuad> weights;
    for (int i = 0; i < 4; ++i)
      weights[i] = .0063971477799023213214514203351730;
    for (int i = 4; i < 8; ++i)
      weights[i] = .0401904480209661724881611584798178;
    for (int i = 8; i < 12; ++i)
      weights[i] = .0243079755047703211748691087719226;
    for (int i = 12; i < 16; ++i)
      weights[i] = .0548588924136974404669241239903914;
    for (int i = 16; i < 22; ++i)
      weights[i] = .0357196122340991824649509689966176;
    for (int i = 22; i < 34; ++i)
      weights[i] = .0071831906978525394094511052198038;
    for (int i = 34; i < 46; ++i)
      weights[i] = .0163721819453191175409381397561191;
    for (int i = 0; i < 46; ++i)
      weights[i] /= 6.0;
    return weights;
  }
};

}  // namespace integrator
}  // namespace mini

#endif  // MINI_INTEGRATOR_TETRA_HPP_
