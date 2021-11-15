// Copyright 2019 Weicheng Pei and Minghao Yang
#ifndef MINI_RIEMANN_ROTATED_SIMPLE_HPP_
#define MINI_RIEMANN_ROTATED_SIMPLE_HPP_

#include "mini/algebra/column.hpp"
#include "mini/algebra/eigen.hpp"

namespace mini {
namespace riemann {
namespace rotated {

template <class UnrotatedSimple>
class Simple {
  using Base = UnrotatedSimple;

 public:
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using State = algebra::Matrix<double, 1, 1>;
  using Flux = algebra::Matrix<double, 1, 1>;
  using Jacobi = typename Base::Jacobi;
  using Coefficient = typename Base::Coefficient;
  void Rotate(Vector const& normal) {
    static_assert(Base::kDim == 2);
    Rotate(normal[0], normal[1]);
  }
  void Rotate(Scalar const& n_1, Scalar const& n_2) {
    static_assert(Base::kDim == 2);
    auto a_normal = global_coefficient[0] * n_1;
    a_normal += global_coefficient[1] * n_2;
    unrotated_simple_ = UnrotatedSimple(a_normal);
  }
  void Rotate(const mini::algebra::Matrix<Scalar, 3, 3> &frame) {
    static_assert(Base::kDim == 3);
    auto &nu = frame.col(0);
    auto a_normal = global_coefficient[0] * nu[0];
    a_normal += global_coefficient[1] * nu[1];
    a_normal += global_coefficient[2] * nu[2];
    unrotated_simple_ = UnrotatedSimple(a_normal);
  }
  Flux GetFluxOnTimeAxis(State const& left, State const& right) {
    Flux flux;
    flux << unrotated_simple_.GetFluxOnTimeAxis(left[0], right[0]);
    return flux;
  }
  Flux GetFluxOnSolidWall(State const& state) {
    Flux flux;
    flux << 0.0;
    return flux;
  }
  Flux GetFluxOnFreeWall(State const& state) {
    Flux flux;
    flux << unrotated_simple_.GetFlux(state[0]);
    return flux;
  }
  static Coefficient global_coefficient;

 private:
  UnrotatedSimple unrotated_simple_;
};
template <class UnrotatedSimple>
typename Simple<UnrotatedSimple>::Coefficient
Simple<UnrotatedSimple>::global_coefficient;

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

#endif  // MINI_RIEMANN_ROTATED_SIMPLE_HPP_
