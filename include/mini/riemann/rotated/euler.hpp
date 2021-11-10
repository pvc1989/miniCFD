//  Copyright 2021 PEI Weicheng and YANG Minghao and JIANG Yuyan

#ifndef MINI_RIEMANN_ROTATED_EULER_HPP_
#define MINI_RIEMANN_ROTATED_EULER_HPP_

#include <initializer_list>
#include <type_traits>

#include "mini/algebra/column.hpp"
#include "mini/algebra/eigen.hpp"

namespace mini {
namespace riemann {
namespace rotated {

template <typename Scalar = double, int kDim = 3>
class Cartesian;

template <typename Scalar>
class Cartesian<Scalar, 2> {
 public:
  using Vector = mini::algebra::Column<Scalar, 2>;

  void Rotate(const Vector& normal) {
    normal_ = normal;
  }
  void Rotate(const Scalar& n_0, const Scalar& n_1) {
    normal_[0] = n_0;
    normal_[1] = n_1;
  }
  void GlobalToNormal(Vector* v) {
    auto& n = normal_;
    /* Calculate the normal component: */
    auto v_n = (*v)[0] * n[0] + (*v)[1] * n[1];
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
  Vector normal_{1.0, 0.0};
  static_assert(std::is_scalar_v<Scalar>);
};

template <typename Scalar>
class Cartesian<Scalar, 3> {
 public:
  using Vector = mini::algebra::Column<Scalar, 3>;

  void Rotate(const Vector& nu, const Vector& sigma) {
    nu_ = nu; sigma_ = sigma;
    // \vec{\pi} = \vec{\nu} \cross \vec{\sigma}
    constexpr int x = 0, y = 1, z = 2;
    pi_[x] = nu[y] * sigma[z] - nu[z] * sigma[y];
    pi_[y] = nu[z] * sigma[x] - nu[x] * sigma[z];
    pi_[z] = nu[x] * sigma[y] - nu[y] * sigma[x];
  }
  void Rotate(const Vector& nu, const Vector& sigma, const Vector& pi) {
    nu_ = nu; sigma_ = sigma; pi_ = pi;
  }
  void Rotate(const mini::algebra::Matrix<Scalar, 3, 3> &frame) {
    auto &nu = frame.col(0), &sigma = frame.col(1), &pi = frame.col(2);
    constexpr int x = 0, y = 1, z = 2;
    nu_[x] = nu[x]; sigma_[x] = sigma[x]; pi_[x] = pi[x];
    nu_[y] = nu[y]; sigma_[y] = sigma[y]; pi_[y] = pi[y];
    nu_[z] = nu[z]; sigma_[z] = sigma[z]; pi_[z] = pi[z];
  }
  void GlobalToNormal(Vector* v) {
    auto v_nu = v->Dot(nu_), v_sigma = v->Dot(sigma_), v_pi = v->Dot(pi_);
    (*v)[0] = v_nu; (*v)[1] = v_sigma; (*v)[2] = v_pi;
  }
  void NormalToGlobal(Vector* v) {
    constexpr int nu = 0, sigma = 1, pi = 2;
    constexpr int  x = 0,     y = 1,  z = 2;
    auto v_x = (*v)[nu] * nu_[x] + (*v)[sigma] * sigma_[x] + (*v)[pi] * pi_[x];
    auto v_y = (*v)[nu] * nu_[y] + (*v)[sigma] * sigma_[y] + (*v)[pi] * pi_[y];
    auto v_z = (*v)[nu] * nu_[z] + (*v)[sigma] * sigma_[z] + (*v)[pi] * pi_[z];
    (*v)[x] = v_x; (*v)[y] = v_y; (*v)[z] = v_z;
  }

 private:
  Vector nu_{1, 0, 0}, sigma_{0, 1, 0}, pi_{0, 0, 1};
  static_assert(std::is_scalar_v<Scalar>);
};

template <class UnrotatedEuler, int kDim = 2>
class Euler {
  using Base = UnrotatedEuler;

 public:
  using Gas = typename Base::Gas;
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using Flux = typename Base::Flux;
  using Conservative = typename Base::Conservative;
  using Primitive = typename Base::Primitive;
  using State = Primitive;

  void Rotate(const Vector& normal) {
    static_assert(kDim == 2);
    cartesian_.Rotate(normal);
  }
  void Rotate(const Scalar& n_1, const Scalar& n_2) {
    static_assert(kDim == 2);
    cartesian_.Rotate(n_1, n_2);
  }
  void Rotate(const Vector& nu, const Vector& sigma) {
    static_assert(kDim == 3);
    cartesian_.Rotate(nu, sigma);
  }
  void Rotate(const Vector& nu, const Vector& sigma, const Vector& pi) {
    static_assert(kDim == 3);
    cartesian_.Rotate(nu, sigma, pi);
  }
  void Rotate(const mini::algebra::Matrix<Scalar, 3, 3>& frame) {
    static_assert(kDim == 3);
    cartesian_.Rotate(frame);
  }
  static Flux GetFlux(State const& state) {
    return Base::GetFlux(state);
  }
  Flux GetFluxOnTimeAxis(
      Conservative const& left,
      Conservative const& right) {
    auto left__primitive = Gas::ConservativeToPrimitive(left);
    auto right_primitive = Gas::ConservativeToPrimitive(right);
    GlobalToNormal(&(left__primitive.momentum));
    GlobalToNormal(&(right_primitive.momentum));
    auto flux = unrotated_euler_.GetFluxOnTimeAxis(
        left__primitive, right_primitive);
    cartesian_.NormalToGlobal(&(flux.momentum));
    return flux;
  }
  Flux GetFluxOnSolidWall(Conservative const& conservative) {
    auto primitive = Gas::ConservativeToPrimitive(conservative);
    auto flux = Flux();
    flux.momentum[0] = primitive.p();
    cartesian_.NormalToGlobal(&(flux.momentum));
    return flux;
  }
  Flux GetFluxOnFreeWall(Conservative const& conservative) {
    auto primitive = Gas::ConservativeToPrimitive(conservative);
    cartesian_.GlobalToNormal(&(primitive.momentum));
    auto flux = unrotated_euler_.GetFlux(primitive);
    cartesian_.NormalToGlobal(&(flux.momentum));
    return flux;
  }
  void GlobalToNormal(Vector* v) {
    cartesian_.GlobalToNormal(v);
  }
  void NormalToGlobal(Vector* v) {
    cartesian_.NormalToGlobal(v);
  }

 private:
  UnrotatedEuler unrotated_euler_;
  Cartesian<Scalar, kDim> cartesian_;
};

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

#endif  //  MINI_RIEMANN_ROTATED_EULER_HPP_
