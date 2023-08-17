//  Copyright 2021 PEI Weicheng and JIANG Yuyan
#ifndef MINI_GAUSS_LINE_HPP_
#define MINI_GAUSS_LINE_HPP_

#include <cmath>
#include <type_traits>

#include "mini/algebra/eigen.hpp"
#include "mini/gauss/gauss.hpp"

namespace mini {
namespace gauss {

template <typename S, int D, int Q>
class Line {
 private:
  using MatDx1 = algebra::Matrix<S, D, 1>;

 public:
  using Scalar = S;
  using LocalCoord = Scalar;
  using GlobalCoord = std::conditional_t<(D > 1), MatDx1, Scalar>;

 private:
  using MatDx2 = algebra::Vector<GlobalCoord, 2>;
  using Mat2x1 = algebra::Vector<Scalar, 2>;
  using Gauss = GaussLegendre<Scalar, Q>;

 private:
  static const std::array<Scalar, Q> local_weights_;
  static const std::array<LocalCoord, Q> local_coords_;
  std::array<Scalar, Q> global_weights_;
  std::array<GlobalCoord, Q> global_coords_;
  MatDx2 pq_;

 public:
  static int CountCorners() {
    return 2;
  }
  static int CountQuadraturePoints() {
    return Q;
  }
  GlobalCoord GetVertex(int i) const {
    return pq_[i];
  }

 private:
  static Scalar Norm(Scalar v) {
    return std::abs(v);
  }
  static Scalar Norm(MatDx1 const &v) {
    return v.norm();
  }
  void BuildQuadraturePoints() {
    GlobalCoord pq = pq_[0] - pq_[1];
    auto det_j = Norm(pq) * 0.5;
    int n = CountQuadraturePoints();
    for (int i = 0; i < n; ++i) {
      global_weights_[i] = local_weights_[i] * det_j;
      global_coords_[i] = LocalToGlobal(GetLocalCoord(i));
    }
  }
  static constexpr auto BuildLocalCoords() {
    return Gauss::points;
  }
  static constexpr auto BuildLocalWeights() {
    return Gauss::weights;
  }
  static Mat2x1 shape_2x1(Scalar x) {
    x *= 0.5;
    return { 0.5 - x, 0.5 + x };
  }

 public:
  Line(GlobalCoord const &p0, GlobalCoord const &p1) {
    pq_[0] = p0; pq_[1] = p1;
    BuildQuadraturePoints();
  }
  GlobalCoord const &GetGlobalCoord(int i) const {
    return global_coords_[i];
  }
  Scalar const &GetGlobalWeight(int i) const {
    return global_weights_[i];
  }
  LocalCoord const &GetLocalCoord(int i) const {
    return local_coords_[i];
  }
  Scalar const &GetLocalWeight(int i) const {
    return local_weights_[i];
  }
  GlobalCoord LocalToGlobal(Scalar x) const {
    return pq_.dot(shape_2x1(x));
  }
};

template <typename S, int D, int Q>
std::array<typename Line<S, D, Q>::LocalCoord, Q> const
Line<S, D, Q>::local_coords_ = Line<S, D, Q>::BuildLocalCoords();

template <typename S, int D, int Q>
std::array<S, Q> const
Line<S, D, Q>::local_weights_ = Line<S, D, Q>::BuildLocalWeights();

}  // namespace gauss
}  // namespace mini

#endif  // MINI_GAUSS_LINE_HPP_
