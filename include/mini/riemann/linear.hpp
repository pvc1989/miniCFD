//  Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_RIEMANN_LINEAR_HPP_
#define MINI_RIEMANN_LINEAR_HPP_

#include <cmath>
#include <array>

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
  using Matrix = std::array<Column, kWaves>;
  // Constructor:
  explicit MultiWave(Matrix const& a_const) : a_const_(a_const) { Decompose(); }
  // Get F on T Axia
  State GetFluxOnTimeAxis(State u_l, State u_r) {
    SetInitial(u_l, u_r);
    return GetFlux(GetState(/* x = */0.0, /* t = */1.0));
  }
  // Get F of U
  State GetFlux(State state) {
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
  // Set U_l and U_r
  void SetInitial(State u_l, State u_r) {
    u_l_ = u_l;
    u_r_ = u_r;
    InitializeV(u_l, u_r);
  }
  // Get U at (x, t)
  State GetState(double x, double t) {
    if (t <= 0) {
      if (x <= 0) {
        return u_l_;
      } else {
        return u_r_;
      }
    } else {
      if (x / t <= eigen_values_[0]) {
        return u_l_;
      } else if (x / t >= eigen_values_[1]) {
        return u_r_;
      } else {
        State v = {v_r_[0], v_l_[1]};
        State u = Dot(positive_matrix_, v);
        return u;
      }
    }
  }

  void Decompose() {
    GetEigenValues();
    GetEigenVectors();
    GetInverseEigenVectors();
  }
  void InitializeV(State u_l, State u_r) {
    //  std::cout << u_l[0] << " " << u_r[1] << std::endl;
    v_l_ = Dot(negative_matrix_, u_l);
    v_r_ = Dot(negative_matrix_, u_r);
  }
  void GetEigenValues() {
    double b = a_const_[0][0] + a_const_[1][1];
    double c = a_const_[0][0] * a_const_[1][1] -
               a_const_[0][1] * a_const_[1][0];
    double delta = std::sqrt(b * b - 4 * c);
    eigen_values_ = {(b - delta) / 2, (b + delta) / 2};
  }
  void GetEigenVectors() {
    positive_matrix_ = {-a_const_[0][1],
                        -a_const_[0][1],
                         a_const_[0][0] - eigen_values_[0],
                         a_const_[0][0] - eigen_values_[1]};
  }

  void GetInverseEigenVectors() {
    double det = positive_matrix_[0][0] * positive_matrix_[1][1] -
                 positive_matrix_[0][1] * positive_matrix_[1][0];
    negative_matrix_ = { positive_matrix_[1][1] / det,
                        -positive_matrix_[0][1] / det,
                        -positive_matrix_[1][0] / det,
                         positive_matrix_[0][0] / det};
  }

 private:
  State u_l_;
  State u_r_;
  State v_l_;
  State v_r_;
  Matrix a_const_;
  Column eigen_values_;
  Matrix positive_matrix_;
  Matrix negative_matrix_;
};


}  //  namespace riemann
}  //  namespace mini

#endif  //  MINI_RIEMANN_LINEAR_HPP_
