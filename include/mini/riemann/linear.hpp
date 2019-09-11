//  Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_RIEMANN_LINEAR_HPP_
#define MINI_RIEMANN_LINEAR_HPP_

#include <cmath>
#include <array>

#include <iostream>

namespace mini {
namespace riemann {

template <class State>
class Flux {
 public:
  Flux GetFlux(State state);
};

template <class State>
class Problem {
 public:
  Flux<State> FluxOnTimeAxis(State state_left, State state_right);
};

class SingleWave {
 public:
  // Types:
  using State = double;
  using Flux = double;
  using Speed = double;
  // Constructor:
  explicit SingleWave(Speed a_const) : a_const_(a_const) {}
  // Get F on T Axia
  Flux GetFluxOnTimeAxis(State u_l, State u_r) {
    SetInitial(u_l, u_r);
    return GetFlux(GetState(/* x = */0.0, /* t = */1.0));
  }
  // Get F of U
  Flux GetFlux(State state) { return state * a_const_; }

 private:
  // Set U_l and U_r
  void SetInitial(State u_l, State u_r) {
    u_l_ = u_l;
    u_r_ = u_r;
  }
  // Get U at (x, t)
  double GetState(double x, double t) {
    if (t <= 0) {
      if (x <= 0) {
        return u_l_;
      } else {
        return u_r_;
      }
    } else {
      if (x / t <= a_const_) {
        return u_l_;
      } else {
        return u_r_;
      }
    }
  }

 private:
  State u_l_, u_r_;
  Speed a_const_;
};

template <int kWaves = 2>
class MultiWave {
 public:
  using State = std::array<double, kWaves>;
  using Flux = State;
  using Column = State;
  using Row = State;
  using Matrix = std::array<Column, kWaves>;
  // Constructor:
  explicit MultiWave(Matrix const& a_const) : a_const_(a_const) { Decompose(); }
  // Get F on T Axia
  State GetFluxOnTimeAxis(State u_l, State u_r) {
    // std::cout << eigen_matrix_l_[0][0] << " " << eigen_matrix_l_[0][1] << std::endl;
    // std::cout << eigen_matrix_l_[1][0] << " " << eigen_matrix_l_[1][1] << std::endl;
    // std::cout << eigen_matrix_r_[0][0] << " " << eigen_matrix_r_[0][1] << std::endl;
    // std::cout << eigen_matrix_r_[1][0] << " " << eigen_matrix_r_[1][1] << std::endl;
    Flux flux;
    if (0 <= eigen_values_[0]) {
      flux = GetFlux(u_l);
    }
    else if (0 >= eigen_values_[1]) {
      flux = GetFlux(u_r);
    } else {
      flux = FluxInsideSector(u_l, u_r, 1);
    }
    return flux;
  }
  // Get F of U
  Flux GetFlux(State state) {
    Flux flux = Dot(a_const_, state);
    return flux;
  }

 private:
  Column Dot(Matrix const& m, Column const& c) {
    auto result = Column();
    for (int i = 0; i < kWaves; i++) {
      for (int j = 0; j < kWaves; j++) {
        result[i] += c[j] * m[i][j];
      }
    }
    return result;
  }
  double Dot(Row const& l, Column const& r) {
    double result = 0.0;
    for (int i = 0; i < kWaves; i++) {
      result += l[i] * r[i];
    }
    return result;
  }
  State FluxInsideSector(State u_l, State u_r, int k) {
    Flux flux;
    for (int i = 0; i < k; i++) {
      Row l = {eigen_matrix_l_[i][0], eigen_matrix_l_[i][1]};
      double temp = Dot(l, u_r) * eigen_values_[i];
      flux[0] += temp * eigen_matrix_r_[0][i];
      flux[1] += temp * eigen_matrix_r_[1][i];
    }
    for (int i = k; i < kWaves; i++) {
      Row l = {eigen_matrix_l_[i][0], eigen_matrix_l_[i][1]};
      double temp = Dot(l, u_l) * eigen_values_[i];
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
    eigen_matrix_l_ = { eigen_matrix_r_[1][1] / det,
                       -eigen_matrix_r_[0][1] / det,
                       -eigen_matrix_r_[1][0] / det,
                        eigen_matrix_r_[0][0] / det};
  }

 private:
  Matrix a_const_;
  Column eigen_values_;
  Matrix eigen_matrix_r_;
  Matrix eigen_matrix_l_;
};


}  //  namespace riemann
}  //  namespace mini

#endif  //  MINI_RIEMANN_LINEAR_HPP_
