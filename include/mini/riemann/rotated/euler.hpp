//  Copyright 2021 PEI Weicheng and YANG Minghao and JIANG Yuyan

#ifndef MINI_RIEMANN_ROTATED_EULER_HPP_
#define MINI_RIEMANN_ROTATED_EULER_HPP_

#include <type_traits>
#include <utility>

#include "mini/algebra/eigen.hpp"
#include "mini/riemann/euler/eigen.hpp"

namespace mini {
namespace riemann {
namespace rotated {

template <typename Scalar = double, int kDimensions = 3>
class Cartesian;

template <typename Scalar>
class Cartesian<Scalar, 2> {
  static constexpr int x{0}, y{1}, n{0}, t{1};

 public:
  using Vector = mini::algebra::Vector<Scalar, 2>;

  void Rotate(const Vector& nu) {
    nu_ = nu;
  }
  void Rotate(Scalar nu_x, Scalar nu_y) {
    nu_[x] = nu_x;
    nu_[y] = nu_y;
  }
  void GlobalToNormal(Vector* v) const {
    /* Calculate the normal component: */
    auto v_n = (*v)[x] * nu_[x] + (*v)[y] * nu_[y];
    /* Calculate the tangential component: */
    auto t_x = -nu_[y], t_y = nu_[x];
    (*v)[t] =  t_x * (*v)[x] + t_y * (*v)[y];
    /* Write the normal component: */
    (*v)[n] = v_n;
  }
  void NormalToGlobal(Vector* v) const {
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
  using Vector = mini::algebra::Vector<Scalar, 3>;

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
  void GlobalToNormal(Vector* v) const {
    auto v_nu = v->dot(nu_), v_sigma = v->dot(sigma_), v_pi = v->dot(pi_);
    constexpr int nu = 0, sigma = 1, pi = 2;
    (*v)[nu] = v_nu; (*v)[sigma] = v_sigma; (*v)[pi] = v_pi;
  }
  void NormalToGlobal(Vector* v) const {
    constexpr int nu = 0, sigma = 1, pi = 2;
    auto v_x = (*v)[nu] * nu_[x] + (*v)[sigma] * sigma_[x] + (*v)[pi] * pi_[x];
    auto v_y = (*v)[nu] * nu_[y] + (*v)[sigma] * sigma_[y] + (*v)[pi] * pi_[y];
    auto v_z = (*v)[nu] * nu_[z] + (*v)[sigma] * sigma_[z] + (*v)[pi] * pi_[z];
    (*v)[x] = v_x; (*v)[y] = v_y; (*v)[z] = v_z;
  }

 private:
  Vector nu_{1, 0, 0}, sigma_{0, 1, 0}, pi_{0, 0, 1};
  static_assert(std::is_scalar_v<Scalar>);

 public:
  const Vector& nu() const {
    return nu_;
  }
  const Vector& sigma() const {
    return sigma_;
  }
  const Vector& pi() const {
    return pi_;
  }
};

template <class UnrotatedEuler>
class Euler {
  using Base = UnrotatedEuler;

 public:
  constexpr static int kComponents = Base::kComponents;
  constexpr static int kDimensions = Base::kDimensions;
  using Gas = typename Base::Gas;
  using Scalar = typename Base::Scalar;
  using Vector = typename Base::Vector;
  using Flux = typename Base::Flux;
  using FluxMatrix = typename Flux::FluxMatrix;
  using Conservative = typename Base::Conservative;
  using Primitive = typename Base::Primitive;

  template <typename... Args>
  void Rotate(Args&&... args) {
    cartesian_.Rotate(std::forward<Args>(args)...);
  }

  static Flux GetFlux(const Primitive& state) {
    return Base::GetFlux(state);
  }
  static FluxMatrix GetFluxMatrix(Conservative const& conservative) {
    return Gas::GetFluxMatrix(conservative);
  }
  Flux GetFluxOnSupersonicInlet(Conservative const& conservative) const {
    auto primitive = Gas::ConservativeToPrimitive(conservative);
    GlobalToNormal(&(primitive.momentum()));
    auto flux = unrotated_euler_.GetFlux(primitive);
    cartesian_.NormalToGlobal(&(flux.momentum()));
    return flux;
  }
  Flux GetFluxOnTimeAxis(Conservative const& left,
      Conservative const& right) const {
    auto left__primitive = Gas::ConservativeToPrimitive(left);
    auto right_primitive = Gas::ConservativeToPrimitive(right);
    GlobalToNormal(&(left__primitive.momentum()));
    GlobalToNormal(&(right_primitive.momentum()));
    auto flux = unrotated_euler_.GetFluxOnTimeAxis(
        left__primitive, right_primitive);
    cartesian_.NormalToGlobal(&(flux.momentum()));
    return flux;
  }
  Flux GetFluxOnSolidWall(Conservative const& conservative) const {
    auto primitive = Gas::ConservativeToPrimitive(conservative);
    Flux flux; flux.setZero();
    flux.momentumX() = primitive.p();
    cartesian_.NormalToGlobal(&(flux.momentum()));
    return flux;
  }
  Flux GetFluxOnSupersonicOutlet(Conservative const& conservative) const {
    auto primitive = Gas::ConservativeToPrimitive(conservative);
    cartesian_.GlobalToNormal(&(primitive.momentum()));
    auto flux = unrotated_euler_.GetFlux(primitive);
    cartesian_.NormalToGlobal(&(flux.momentum()));
    return flux;
  }
  Flux GetFluxOnSubsonicInlet(Conservative const& conservative_i,
      Conservative const& conservative_o) const {
    auto primitive_i = Gas::ConservativeToPrimitive(conservative_i);
    auto primitive_o = Gas::ConservativeToPrimitive(conservative_o);
    Primitive primitive_b = primitive_o;
    Scalar u_nu_o = primitive_o.momentum().dot(cartesian_.nu());
    Scalar u_nu_i = primitive_i.momentum().dot(cartesian_.nu());
    Scalar u_nu_jump = u_nu_o - u_nu_i;
    Scalar a_i = Gas::GetSpeedOfSound(primitive_i);
    Scalar rho_a_i = primitive_i.rho() * (u_nu_o > 0 ? a_i : -a_i);
    primitive_b.p() = (primitive_i.p() + primitive_o.p()
        + rho_a_i * u_nu_jump) * 0.5;
    Scalar p_jump = primitive_o.p() - primitive_b.p();
    primitive_b.rho() -= p_jump / (a_i * a_i);
    primitive_b.momentum() += (p_jump / rho_a_i) * cartesian_.nu();
    GlobalToNormal(&(primitive_b.momentum()));
    auto flux = unrotated_euler_.GetFlux(primitive_b);
    NormalToGlobal(&(flux.momentum()));
    return flux;
  }
  Flux GetFluxOnSubsonicOutlet(Conservative const& conservative_i,
      Conservative const& conservative_o) const {
    auto primitive_i = Gas::ConservativeToPrimitive(conservative_i);
    auto primitive_o = Gas::ConservativeToPrimitive(conservative_o);
    Primitive primitive_b = primitive_i;
    primitive_b.p() = primitive_o.p();
    Scalar p_jump = primitive_i.p() - primitive_b.p();
    Scalar a_i = Gas::GetSpeedOfSound(primitive_i);
    primitive_b.rho() -= p_jump / (a_i * a_i);
    Scalar u_nu_i = primitive_i.momentum().dot(cartesian_.nu());
    Scalar rho_a_i = primitive_i.rho() * (u_nu_i > 0 ? a_i : -a_i);
    primitive_b.momentum() += (p_jump / rho_a_i) * cartesian_.nu();
    GlobalToNormal(&(primitive_b.momentum()));
    auto flux = unrotated_euler_.GetFlux(primitive_b);
    NormalToGlobal(&(flux.momentum()));
    return flux;
  }
  Flux GetFluxOnSmartBoundary(Conservative const& conservative_i,
      Conservative const& conservative_o) const {
    auto primitive = Gas::ConservativeToPrimitive(conservative_i);
    Scalar a = Gas::GetSpeedOfSound(primitive);
    Scalar u_nu = primitive.momentum().dot(cartesian_.nu());
    Flux flux;
    if (u_nu < 0) {  // inlet
      if (u_nu + a < 0) {
        flux = GetFluxOnSupersonicInlet(conservative_o);
      } else {
        flux = GetFluxOnSubsonicInlet(conservative_i, conservative_o);
      }
    } else {  // outlet
      if (u_nu - a > 0) {
        flux = GetFluxOnSupersonicOutlet(conservative_i);
      } else {
        flux = GetFluxOnSubsonicOutlet(conservative_i, conservative_o);
      }
    }
    return flux;
  }
  void GlobalToNormal(Vector* v) const {
    cartesian_.GlobalToNormal(v);
  }
  void NormalToGlobal(Vector* v) const {
    cartesian_.NormalToGlobal(v);
  }

 private:
  using EigenMatrices = riemann::euler::EigenMatrices<Gas>;
  EigenMatrices eigen_matrices_;
  UnrotatedEuler unrotated_euler_;
  Cartesian<Scalar, kDimensions> cartesian_;

 public:
  using Matrix = typename EigenMatrices::Mat5x5;
  void UpdateEigenMatrices(const Conservative& big_u) {
    eigen_matrices_ = EigenMatrices(big_u,
        cartesian_.nu(), cartesian_.sigma(), cartesian_.pi());
  }
  const Matrix& L() const {
    return eigen_matrices_.L;
  }
  const Matrix& R() const {
    return eigen_matrices_.R;
  }
};

}  // namespace rotated
}  // namespace riemann
}  // namespace mini

#endif  //  MINI_RIEMANN_ROTATED_EULER_HPP_
