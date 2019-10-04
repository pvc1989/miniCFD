//  Copyright 2019 Weicheng Pei and Minghao Yang

#ifndef MINI_RIEMANN_ROTATED_EULER_HPP_
#define MINI_RIEMANN_ROTATED_EULER_HPP_

#include <initializer_list>

#include "mini/algebra/column.hpp"
#include "mini/algebra/matrix.hpp"

namespace mini {
namespace riemann {
namespace rotated {

template <class UnrotatedEuler>
class Euler {
  using Base = UnrotatedEuler;

 public:
  using Gas = typename Base::Gas;
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using Conservative = typename Base::Conservative;
  using Primitive = typename Base::Primitive;
  using State = Conservative;
  using Flux = typename Base::Flux;
  void Rotate(Vector const& normal) { normal_ = normal; }
  void Rotate(Scalar const& n_1, Scalar const& n_2) {
    normal_[0] = n_1;
    normal_[1] = n_2;
  }
  Flux GetFluxOnTimeAxis(Conservative const& left, Conservative const& right) {
    auto left__primitive = Gas::ConservativeToPrimitive(left);
    auto right_primitive = Gas::ConservativeToPrimitive(right);
    GlobalToNormal(&(left__primitive.momentum));
    GlobalToNormal(&(right_primitive.momentum));
    auto flux = unrotated_euler_.GetFluxOnTimeAxis(
        left__primitive, right_primitive);
    NormalToGlobal(&(flux.momentum));
    return flux;
  }
  Flux GetFluxOnSolidWall(Conservative const& conservative) {
    auto primitive = Gas::ConservativeToPrimitive(conservative);
    auto flux = Flux();
    flux.momentum[0] = primitive.p();
    NormalToGlobal(&(flux.momentum));
    return flux;
  }
  Flux GetFluxOnFreeWall(Conservative const& conservative) {
    auto primitive = Gas::ConservativeToPrimitive(conservative);
    GlobalToNormal(&(primitive.momentum));
    auto flux = unrotated_euler_.GetFlux(primitive);
    NormalToGlobal(&(flux.momentum));
    return flux;
  }
  void GlobalToNormal(Vector* v) {
    auto& n = normal_;
    /* Calculate the normal component: */
    auto v_n = v->Dot(n);
    /* Calculate the tangential component:
       auto t = Vector{ -n[1], n[0] };
       auto v_t = v->Dot(t);
    */
    (*v)[1] = n[0] * (*v)[1] - n[1] * (*v)[0];
    /* Write the normal component: */
    (*v)[0] = v_n;
  }
  void NormalToGlobal(Vector* v) {
    auto& n = normal_;
    auto v_0 = (*v)[0] * n[0] - (*v)[1] * n[1];
    (*v)[1] = (*v)[0] * n[1] + (*v)[1] * n[0];
    (*v)[0] = v_0;
  }

 private:
  UnrotatedEuler unrotated_euler_;
  Vector normal_;
};

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

#endif  //  MINI_RIEMANN_ROTATED_EULER_HPP_
