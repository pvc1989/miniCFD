//  Copyright 2021 PEI Weicheng and YANG Minghao and JIANG Yuyan

#ifndef MINI_RIEMANN_ROTATED_EULER_HPP_
#define MINI_RIEMANN_ROTATED_EULER_HPP_

#include <initializer_list>
#include <type_traits>
#include <utility>

#include "mini/algebra/column.hpp"
#include "mini/algebra/eigen.hpp"

namespace mini {
namespace riemann {
namespace rotated {

template <typename Scalar = double, int kDim = 3>
class Cartesian;

template <typename Scalar>
class Cartesian<Scalar, 2> {
  static constexpr int x{0}, y{1}, n{0}, t{1};

 public:
  using Vector = mini::algebra::Column<Scalar, 2>;

  void Rotate(const Vector& nu) {
    nu_ = nu;
  }
  void Rotate(const Scalar& nu_x, const Scalar& nu_y) {
    nu_[x] = nu_x;
    nu_[y] = nu_y;
  }
  void GlobalToNormal(Vector* v) {
    /* Calculate the normal component: */
    auto v_n = (*v)[x] * nu_[x] + (*v)[y] * nu_[y];
    /* Calculate the tangential component: */
    auto t_x = -nu_[y], t_y = nu_[x];
    (*v)[t] =  t_x * (*v)[x] + t_y * (*v)[y];
    /* Write the normal component: */
    (*v)[n] = v_n;
  }
  void NormalToGlobal(Vector* v) {
    auto t_x = -nu_[y], t_y = nu_[x];
    auto v_x = (*v)[n] * nu_[x] + (*v)[t] * t_x;
    (*v)[y]  = (*v)[n] * nu_[y] + (*v)[t] * t_y;
    (*v)[x] = v_x;
  }

 private:
  Vector nu_{1.0, 0.0};
  static_assert(std::is_scalar_v<Scalar>);
};

template <typename Scalar>
class Cartesian<Scalar, 3> {
  static constexpr int x = 0, y = 1, z = 2;

 public:
  using Vector = mini::algebra::Column<Scalar, 3>;

  void Rotate(const Vector& nu, const Vector& sigma) {
    nu_ = nu; sigma_ = sigma;
    // \vec{\pi} = \vec{\nu} \cross \vec{\sigma}
    pi_[x] = nu[y] * sigma[z] - nu[z] * sigma[y];
    pi_[y] = nu[z] * sigma[x] - nu[x] * sigma[z];
    pi_[z] = nu[x] * sigma[y] - nu[y] * sigma[x];
  }
  void Rotate(const Vector& nu, const Vector& sigma, const Vector& pi) {
    nu_ = nu; sigma_ = sigma; pi_ = pi;
  }
  void Rotate(const mini::algebra::Matrix<Scalar, 3, 3> &frame) {
    auto &nu = frame.col(0), &sigma = frame.col(1), &pi = frame.col(2);
    nu_[x] = nu[x]; sigma_[x] = sigma[x]; pi_[x] = pi[x];
    nu_[y] = nu[y]; sigma_[y] = sigma[y]; pi_[y] = pi[y];
    nu_[z] = nu[z]; sigma_[z] = sigma[z]; pi_[z] = pi[z];
  }
  void GlobalToNormal(Vector* v) {
    auto v_nu = v->Dot(nu_), v_sigma = v->Dot(sigma_), v_pi = v->Dot(pi_);
    constexpr int nu = 0, sigma = 1, pi = 2;
    (*v)[nu] = v_nu; (*v)[sigma] = v_sigma; (*v)[pi] = v_pi;
  }
  void NormalToGlobal(Vector* v) {
    constexpr int nu = 0, sigma = 1, pi = 2;
    auto v_x = (*v)[nu] * nu_[x] + (*v)[sigma] * sigma_[x] + (*v)[pi] * pi_[x];
    auto v_y = (*v)[nu] * nu_[y] + (*v)[sigma] * sigma_[y] + (*v)[pi] * pi_[y];
    auto v_z = (*v)[nu] * nu_[z] + (*v)[sigma] * sigma_[z] + (*v)[pi] * pi_[z];
    (*v)[x] = v_x; (*v)[y] = v_y; (*v)[z] = v_z;
  }

 private:
  Vector nu_{1, 0, 0}, sigma_{0, 1, 0}, pi_{0, 0, 1};
  static_assert(std::is_scalar_v<Scalar>);
};

template <class UnrotatedEuler>
class Euler {
  using Base = UnrotatedEuler;
  constexpr static int kDim = Base::kDim;

 public:
  using Gas = typename Base::Gas;
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using Flux = typename Base::Flux;
  using FluxMatrix = typename Flux::FluxMatrix;
  using Conservative = typename Base::Conservative;
  using Primitive = typename Base::Primitive;
  using State = Primitive;

  template <typename... Args>
  void Rotate(Args&&... args) {
    cartesian_.Rotate(std::forward<Args>(args)...);
  }

  static Flux GetFlux(State const& state) {
    return Base::GetFlux(state);
  }
  Flux GetRotatedFlux(Conservative const& conservative) {
    auto primitive = Gas::ConservativeToPrimitive(conservative);
    GlobalToNormal(&(primitive.momentum));
    auto flux = unrotated_euler_.GetFlux(primitive);
    cartesian_.NormalToGlobal(&(flux.momentum));
    return flux;
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
  static FluxMatrix GetFluxMatrix(Conservative const& conservative) {
    return Gas::GetFluxMatrix(conservative);
  }

 private:
  UnrotatedEuler unrotated_euler_;
  Cartesian<Scalar, kDim> cartesian_;
};

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

#endif  //  MINI_RIEMANN_ROTATED_EULER_HPP_
