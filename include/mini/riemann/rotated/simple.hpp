#ifndef MINI_RIEMANN_ROTATED_SIMPLE_HPP_
#define MINI_RIEMANN_ROTATED_SIMPLE_HPP_

#include "mini/algebra/column.hpp"

namespace mini {
namespace riemann {
namespace rotated {

template <class UnrotatedSimple>
class Simple {
  using Base = UnrotatedSimple;

 public:
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using State = typename Base::State;
  using Flux = typename Base::Flux;
  using Jacobi = typename Base::Jacobi;
  using Coefficient = algebra::Column<Jacobi, 2>;
  void Rotate(Vector const& normal) {
    Rotate(normal[0], normal[1]);
  }
  void Rotate(Scalar const& n_1, Scalar const& n_2) {
    auto a_normal = global_coefficient[0] * n_1;
    a_normal += global_coefficient[1] * n_2;
    unrotated_simple_= UnrotatedSimple(a_normal);
  }
  Flux GetFluxOnTimeAxis(State const& left, State const& right) {
    auto flux = unrotated_simple_.GetFluxOnTimeAxis(left, right);
    return flux;
  }
  Flux GetFluxOnSolidWall(State const& state) {
    return {};
  }
  Flux GetFluxOnFreeWall(State const& state) {
    return unrotated_simple_.GetFlux(state);
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

#endif  //  MINI_RIEMANN_ROTATED_SIMPLE_HPP_