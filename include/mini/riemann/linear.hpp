//  Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_RIEMANN_LINEAR_HPP_
#define MINI_RIEMANN_LINEAR_HPP_

#include <cmath>
#include <array>

#include "mini/geometry/dim0.hpp"

namespace mini {
namespace riemann {

class SingleWave {
 public:
  // Types:
  using Jacobi = double;
  using State = double;
  using Flux = double;
  using Speed = double;
  // Constructor:
  SingleWave() : a_const_(1) {}
  explicit SingleWave(Jacobi const& a_const) : a_const_(a_const) {}
  // Get F on T Axia
  Flux GetFluxOnTimeAxis(State const& left, State const& right) const {
    if (0 < a_const_) {
      return left * a_const_;
    } else {
      return right* a_const_;
    }
  }
  // Get F of U
  Flux GetFlux(State const& state) const { return state * a_const_ ; }

 private:
  Jacobi a_const_;
};

template <int kWaves = 2>
class MultiWave {
 public:
  using Column = geometry::Vector<double, kWaves>;
  using Row = Column;
  using Matrix = geometry::Vector<Row, kWaves>;
  using Jacobi = Matrix;
  using State = Column;
  using Flux = Column;
  // Constructor:
  MultiWave() = default;
  explicit MultiWave(Jacobi const& a_const) : a_const_(a_const) { Decompose(); }
  // Get F on T Axia
  State GetFluxOnTimeAxis(State const& left, State const& right) const {
    Flux flux;
    if (0 <= eigen_values_[0]) {
      flux = GetFlux(left);
    } else if (0 >= eigen_values_[1]) {
      flux = GetFlux(right);
    } else {
      flux = FluxInsideSector(left, right, 1);
    }
    return flux;
  }
  // Get F of U
  Flux GetFlux(State const& state) const {
    return Dot(a_const_, state);
  }

 private:
  Column Dot(Matrix const& m, Column const& c) const {
    auto result = Column();
    for (int i = 0; i < kWaves; i++) {
      result[i] = m[i].Dot(c);
    }
    return result;
  }
  State FluxInsideSector(State const& left, State const& right, int k) const {
    Flux flux{0, 0};
    for (int i = 0; i < k; i++) {
      Row l = {eigen_matrix_l_[i][0], eigen_matrix_l_[i][1]};
      double temp = l.Dot(right) * eigen_values_[i];
      flux[0] += temp * eigen_matrix_r_[0][i];
      flux[1] += temp * eigen_matrix_r_[1][i];
    }
    for (int i = k; i < kWaves; i++) {
      Row l = {eigen_matrix_l_[i][0], eigen_matrix_l_[i][1]};
      double temp = l.Dot(left) * eigen_values_[i];
      flux[0] += temp * eigen_matrix_r_[0][i];
      flux[1] += temp * eigen_matrix_r_[1][i];
    }
    return flux;
  }
  void Decompose() {
    GetEigenValues();
    GetEigenVectors();
    GetInverseEigenVectors();
  }
  void GetEigenValues() {
    double b = a_const_[0][0] + a_const_[1][1];
    double c = a_const_[0][0] * a_const_[1][1] -
               a_const_[0][1] * a_const_[1][0];
    double delta = std::sqrt(b * b - 4 * c);
    eigen_values_ = {(b - delta) / 2, (b + delta) / 2};
  }
  void GetEigenVectors() {
    double a = a_const_[0][0] - eigen_values_[0];
    double b = a_const_[0][1];
    double c = a_const_[1][0];
    double d = a_const_[1][1] - eigen_values_[0];
    if (a == 0 && b == 0) {
      eigen_matrix_r_[0][0] = d;
      eigen_matrix_r_[1][0] = -c;
    } else {
      eigen_matrix_r_[0][0] = b;
      eigen_matrix_r_[1][0] = -a;
    }
    a = a_const_[0][0] - eigen_values_[1];
    d = a_const_[1][1] - eigen_values_[1];
    if (a == 0 && b == 0) {
      eigen_matrix_r_[0][1] = d;
      eigen_matrix_r_[1][1] = -c;
    } else {
      eigen_matrix_r_[0][1] = b;
      eigen_matrix_r_[1][1] = -a;
    }
  }
  void GetInverseEigenVectors() {
    double det = eigen_matrix_r_[0][0] * eigen_matrix_r_[1][1] -
                 eigen_matrix_r_[0][1] * eigen_matrix_r_[1][0];
    eigen_matrix_l_[0][0] =  eigen_matrix_r_[1][1] / det;
    eigen_matrix_l_[0][1] = -eigen_matrix_r_[0][1] / det;
    eigen_matrix_l_[1][0] = -eigen_matrix_r_[1][0] / det;
    eigen_matrix_l_[1][1] =  eigen_matrix_r_[0][0] / det;
  }

 private:
  Jacobi a_const_;
  Column eigen_values_;
  Matrix eigen_matrix_r_;
  Matrix eigen_matrix_l_;
};

}  //  namespace riemann
}  //  namespace mini

#endif  //  MINI_RIEMANN_LINEAR_HPP_
